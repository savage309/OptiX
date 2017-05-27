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


inline void checkOptErrorContext(const char* file, int line, ResultType result, Context ctx) {
    if (result == 0)
        return;
    const char* returnString;
    rtContextGetErrorString(ctx, RTresult(result), &returnString);
    printf("Erorr 0x%x %s (file %s, line %i)",int(result),returnString, file, line);
    vassert(false);
    
}

inline void checkOptError(const char* file, int line, ResultType result) {
    if (result == 0)
        return;
    printf("Erorr 0x%x (file %s, line %i)", int(result), file, line);
    vassert(false);
}

#define CHECK_ERROR(X) checkOptError(__FILE__, __LINE__, X)
#define CHECK_ERROR_CTX(X) checkOptErrorContext(__FILE__, __LINE__, X, context)


std::string buildPTX() {
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
        return "";
    }
    
    size_t ptxSize;
    nvRes = nvrtcGetPTXSize(program, &ptxSize);
    CHECK_ERROR(nvRes);
    
    std::unique_ptr<char[]> ptx(new char[ptxSize + 1]);
    nvRes = nvrtcGetPTX(program, ptx.get());
    
    return ptx.get();
}

struct GPUProgram {
    void create() {
        CHECK_ERROR_CTX(rtContextCreate(&context));
    }
    
    void destroy() {
        CHECK_ERROR_CTX(rtContextDestroy(context));
    }
    
    void setRayTypeCount(int rayTypeCount) {
        vassert(rayTypeCount > 0);
        CHECK_ERROR_CTX(rtContextSetRayTypeCount(context, rayTypeCount));
    }
    
    void setEntryPointCount(int entryPointCount) {
        vassert(entryPointCount > 0);
        CHECK_ERROR_CTX(rtContextSetEntryPointCount(context, entryPointCount));
    }
    
    void setStackSize(int stackSize) {
        vassert(stackSize > 0);
        CHECK_ERROR_CTX(rtContextSetStackSize(context, stackSize));
    }
    
    void declareVariable(const char* name, Variable* var) {
        vassert(name && var);
        CHECK_ERROR_CTX(rtContextDeclareVariable(context, name, var));
    }
    
    void setRayGenerationProgram(const char* ptx, const char* programName) {
        CHECK_ERROR_CTX(rtContextSetEntryPointCount(context, 1));
        CHECK_ERROR_CTX(rtContextSetRayGenerationProgram(context, 0, getRTProgram(ptx, programName)));
    }
    
    void setExceptionProgram(const char* ptx, const char* programName) {
        CHECK_ERROR_CTX(rtContextSetExceptionProgram(context, 0, getRTProgram(ptx, programName)));
    }
    
private:
    RTprogram getRTProgram(const char* ptx, const char* programName) {
        RTprogram program;
        CHECK_ERROR_CTX(rtProgramCreateFromPTXString(context, ptx, programName, &program));
        return program;
    }
    
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
    
    const std::string& ptxSource = buildPTX();
    
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
    
    program.setRayGenerationProgram(ptxSource.c_str(), "generatePrimaryRay");
    program.setExceptionProgram(ptxSource.c_str(), "exception");
    
    program.destroy();
    
    return 0;
}
