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
typedef RTvariable GPUVariable;
typedef RTmaterial GPUMaterial;
typedef RTcontext GPUContext;

#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))

enum LogType { LogTypeInfo = 0, LogTypeWarning, LogTypeError, LogTypeNone };

const LogType LOG_LEVEL = LogTypeInfo;

void printLog(LogType priority, const char *format, ...) {
    
    if(priority < LOG_LEVEL)
        return;
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

inline void checkOptErrorContext(const char* file, int line, ResultType result, GPUContext ctx) {
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

//wrapper ot (OptiX)Variable
struct Variable {
    void set(unsigned size, const void* hostPtr) {
        CHECK_ERROR(rtVariableSetUserData(variable, size, hostPtr));
    }
    GPUVariable* get() {
        return &variable;
    }
private:
    GPUVariable variable;
};

//wrapper of (OptiX)Context
struct Context {
    Context():context(NULL) {}
    ~Context() {
        vassert(context == NULL);
    }
    
    void init() {
        vassert(context == NULL);
        CHECK_ERROR_CTX(rtContextCreate(&context));
    }
    
    void freeMem() {
        CHECK_ERROR_CTX(rtContextDestroy(context));
        context = NULL;
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
    
    void declareVariable(const char* name, Variable& var) {
        vassert(name && var.get());
        CHECK_ERROR_CTX(rtContextDeclareVariable(context, name, var.get()));
    }
    
    void setRayGenerationProgram(const char* ptx, const char* programName) {
        CHECK_ERROR_CTX(rtContextSetEntryPointCount(context, 1));
        CHECK_ERROR_CTX(rtContextSetRayGenerationProgram(context, 0, getRTProgram(ptx, programName)));
    }
    
    void setExceptionProgram(const char* ptx, const char* programName) {
        CHECK_ERROR_CTX(rtContextSetExceptionProgram(context, 0, getRTProgram(ptx, programName)));
    }
    
    GPUContext get() {
        return context;
    }
    
    RTprogram getRTProgram(const char* ptx, const char* programName) {
        RTprogram program;
        CHECK_ERROR_CTX(rtProgramCreateFromPTXString(context, ptx, programName, &program));
        return program;
    }
    
private:
    GPUContext context;
};

//wrapper of (OptiX)Material
struct Material {
    Material(Context& context):context(context) {
        CHECK_ERROR(rtMaterialCreate(context.get(), &material));
    }
    ~Material() {
        vassert(material==NULL);
    }
    
    void freeMem() {
        vassert(material);
        CHECK_ERROR(rtMaterialDestroy(material));
        material = NULL;
    }
    
    void setClosestHitProgram(const char* ptx, unsigned rayType, const char* programName) {
        RTprogram closestHitProgram = context.getRTProgram(ptx, programName);
        CHECK_ERROR(rtMaterialSetClosestHitProgram(material, rayType, closestHitProgram));
    }
    
    void setAnyHitProgram(const char* ptx, unsigned rayType, const char* programName) {
        RTprogram anyHitProgram = context.getRTProgram(ptx, programName);
        CHECK_ERROR(rtMaterialSetAnyHitProgram(material, rayType, anyHitProgram));
    }
    
    GPUMaterial get() {
        return material;
    }
    
private:
    Context& context;
    GPUMaterial material;
};

enum {
    ResultSuccess = 0,
};

struct clRenderData {
    float dof;
};

int main(int argc, const char* argv[]) {
    const std::string& ptxSource = buildPTX();
    
    CHECK_ERROR(cuInit(0));
    
    Context context;
    context.init();
    
    context.setRayTypeCount(1);
    context.setStackSize(14000);
    context.setRayTypeCount(1);
    
    Variable renderData;
    clRenderData hostRenderData;
    context.declareVariable("renderData", renderData);
    renderData.set(sizeof(clRenderData), &hostRenderData);
    
    context.setRayGenerationProgram(ptxSource.c_str(), "generatePrimaryRay");
    context.setExceptionProgram(ptxSource.c_str(), "exception");
    
    Material material(context);
    material.setClosestHitProgram(ptxSource.c_str(), 0, "materialHit");
    material.setAnyHitProgram(ptxSource.c_str(), 0, "materialMiss");
    
    material.freeMem();
    context.freeMem();
    
    return 0;
}
