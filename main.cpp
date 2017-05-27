#define VASSERT_ENABLED

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>

#include "cuda.h"
#include "nvrtc.h"
#include "optix.h"

inline void vassert(bool b) {
#ifdef VASSERT_ENABLED
    if (!b) {
        int* ptr = NULL;
        *ptr = 196;
    }
#endif //VASSERT_ENABLED
}
typedef int ResultType;
typedef RTcontext Context;
typedef RTvariable Variable;

#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))

enum LogType { LogTypeInfo = 0, LogTypeInfoRaw = 0, LogTypeWarning, LogTypeError, LogTypeNone };

const LogType LOG_LEVEL = LogTypeInfo;

void printLog(LogType priority, const char *format, ...) {
    
    if(priority < LOG_LEVEL)
        return;
    if (priority != LogTypeInfoRaw) {
        char s[512];
        time_t t = time(NULL);
        struct tm * p = localtime(&t);
        strftime(s, 512, "[%H:%M:%S] ", p);
        printf("%s", s);
        switch (priority) {
            case LogTypeInfo:
                printf("Info: ");
                break;
            case LogTypeWarning:
                printf("Warning: ");
                break;
            case LogTypeError:
                printf("Error: ");
                break;
            default:
                break;
        }
    }
    
    va_list args;
    va_start(args, format);
    
    vprintf(format, args);
    
    va_end(args);
}

std::string getProgramSource(const std::string& path) {
    std::ifstream programSource(path.c_str());
    if (!programSource.good()) printLog(LogTypeError, "program source not found\n");
    return std::string((std::istreambuf_iterator <char>(programSource)), std::istreambuf_iterator <char>());
}

inline void checkOptError(const char* file, int line, ResultType result, Context* ctx = NULL) {
    if (result != RT_SUCCESS) {
        if (ctx) {
            const char* returnString;
            rtContextGetErrorString(*ctx, RTresult(result), &returnString);
            printf("Erorr 0x%x %s (file %s, line %i)",int(result),returnString, file, line);
            
        } else
            printf("Erorr 0x%x (file %s, line %i)", int(result), file, line);
        vassert(false);
    }
}

#define CHECK_ERROR(X) checkOptError(__FILE__, __LINE__, X)
#define CHECK_ERROR_CTX(X, CTX) checkOptError(__FILE__, __LINE__, X, CTX)


void buildPTX() {
    const char* pathToKernelSource = "/Developer/git/OptiX/OptiX/kernel.cu";
    printLog(LogTypeInfo, "Trying to load kernel from source located at %s\n", pathToKernelSource);
    std::string source = getProgramSource(pathToKernelSource);
    
    nvrtcResult nvRes;
    nvrtcProgram program;
    nvRes = nvrtcCreateProgram(&program, source.c_str(), "compiled_kernel", 0, NULL, NULL);
    
    const char* options[] = {"--gpu-architecture=compute_20","--maxrregcount=64","--use_fast_math"};//, "--std=c++11"};
    nvRes = nvrtcCompileProgram(program, COUNT_OF(options), options);
    
    if (nvRes != NVRTC_SUCCESS) {
        size_t programLogSize;
        nvRes = nvrtcGetProgramLogSize(program, &programLogSize);
        CHECK_ERROR(nvRes);
        std::unique_ptr<char[]> log (new char[programLogSize + 1]);
        
        nvRes = nvrtcGetProgramLog(program, log.get());
        CHECK_ERROR(nvRes);
        printLog(LogTypeError, "%s", log.get());
        return;
    }
    
    size_t ptxSize;
    nvRes = nvrtcGetPTXSize(program, &ptxSize);
    CHECK_ERROR(nvRes);
    
    std::unique_ptr<char[]> ptx(new char[ptxSize + 1]);
    nvRes = nvrtcGetPTX(program, ptx.get());
    
    const char* TARGET_CUDA_SAVE_PTX_PATH = "/Developer/git/OptiX/OptiX/blago.ptx";
    
    {
        std::fstream ptxStream(TARGET_CUDA_SAVE_PTX_PATH, std::ios_base::trunc | std::ios_base::out);
        if (!ptxStream.good()) {
            printLog(LogTypeWarning, "Could not save PTX IR to %s\n", TARGET_CUDA_SAVE_PTX_PATH);
        } else {
            printLog(LogTypeInfo, "PTX IR saved to %s\n", TARGET_CUDA_SAVE_PTX_PATH);
        }
        ptxStream << ptx.get();
    }
    
    const size_t JIT_NUM_OPTIONS = 9;
    const size_t JIT_BUFFER_SIZE_IN_BYTES = 1024;
    char logBuffer[JIT_BUFFER_SIZE_IN_BYTES];
    char errorBuffer[JIT_BUFFER_SIZE_IN_BYTES];
    
    CUjit_option jitOptions[JIT_NUM_OPTIONS];
    int optionsCounter = 0;
    jitOptions[optionsCounter++] = CU_JIT_MAX_REGISTERS;
    jitOptions[optionsCounter++] = CU_JIT_OPTIMIZATION_LEVEL;
    jitOptions[optionsCounter++] = CU_JIT_TARGET_FROM_CUCONTEXT;
    jitOptions[optionsCounter++] = CU_JIT_FALLBACK_STRATEGY;
    jitOptions[optionsCounter++] = CU_JIT_INFO_LOG_BUFFER;
    jitOptions[optionsCounter++] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    jitOptions[optionsCounter++] = CU_JIT_ERROR_LOG_BUFFER;
    jitOptions[optionsCounter++] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    jitOptions[optionsCounter++] = CU_JIT_GENERATE_LINE_INFO;
    void* jitValues[JIT_NUM_OPTIONS];
    const int maxRegCount = 63;
    int valuesCounter = 0;
    jitValues[valuesCounter++] = (void*)maxRegCount;
    const int optimizationLevel = 4;
    jitValues[valuesCounter++] = (void*)optimizationLevel;
    const int dummy = 0;
    jitValues[valuesCounter++] = (void*)dummy;
    const CUjit_fallback_enum fallbackStrategy = CU_PREFER_PTX;
    jitValues[valuesCounter++] = (void*)fallbackStrategy;
    jitValues[valuesCounter++] = (void*)logBuffer;
    const int logBufferSize = JIT_BUFFER_SIZE_IN_BYTES;
    jitValues[valuesCounter++] = (void*)logBufferSize;
    jitValues[valuesCounter++] = (void*)errorBuffer;
    const int errorBufferSize = JIT_BUFFER_SIZE_IN_BYTES;
    jitValues[valuesCounter++] = (void*)errorBufferSize;
    const int generateLineInfo = 1;
    jitValues[valuesCounter++] = (void*)generateLineInfo;
    
    
    printLog(LogTypeInfo, "PTX %s", ptx.get());
    //for (int i = 0; i < devices.size(); ++i) {
    //    CUmodule program;
    //    CUresult err = cuModuleLoadDataEx(&program, ptx.get(), JIT_NUM_OPTIONS, jitOptions, jitValues);
    //    CHECK_ERROR(err);
    //    programs.push_back(program);
    //    printLog(LogTypeInfo, "program for device %i compiled\n", i);
    //}
    nvRes = nvrtcDestroyProgram(&program);
    CHECK_ERROR(nvRes);
}

struct GPUProgram {
    void create() {
        CHECK_ERROR(rtContextCreate(&context));
    }
    
    void destroy() {
        CHECK_ERROR(rtContextDestroy(context));
    }
    
    void setRayTypeCount(int rayTypeCount) {
        vassert(rayTypeCount > 0);
        CHECK_ERROR(rtContextSetRayTypeCount(context, rayTypeCount));
    }
    
    void setEntryPointCount(int entryPointCount) {
        vassert(entryPointCount > 0);
        CHECK_ERROR(rtContextSetEntryPointCount(context, entryPointCount));
    }
    
    void setStackSize(int stackSize) {
        vassert(stackSize > 0);
        CHECK_ERROR(rtContextSetStackSize(context, stackSize));
    }
    
    void declareVariable(const char* name, Variable* var) {
        vassert(name && var);
        CHECK_ERROR(rtContextDeclareVariable(context, name, var));
    }
    
private:
    Context context;
};

void setVariable(Variable v, unsigned size, const void* hostPtr) {
    CHECK_ERROR(rtVariableSetUserData(v, size, hostPtr));
}

enum {
    ResultSuccess = 0,
};

struct clRenderData {
    float dof;
};

int main(int argc, const char* argv[]) {
    GPUProgram program;
    
    buildPTX();
    
    ResultType err = ResultSuccess;
    err = cuInit(0);
    
    CHECK_ERROR(err);
    
    clRenderData hostRenderData;
    
    Variable renderData;
    
    program.create();
    
    program.setRayTypeCount(1);
    program.setStackSize(14000);
    
    program.declareVariable("renderData", &renderData);
    setVariable(renderData, sizeof(clRenderData), &hostRenderData);
    
    program.destroy();
    
    return 0;
}
