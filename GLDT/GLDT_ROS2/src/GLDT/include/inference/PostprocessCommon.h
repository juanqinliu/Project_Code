#pragma once

#include <cstdlib>
#include "common/Flags.h"  // 添加Flags头文件

namespace tracking {

// 预处理模式枚举
enum class PreprocessMode {
    CPU = 0,            // 使用CPU完整预处理
    CPU_GPU_HYBRID = 1, // 使用CPU+GPU混合预处理
    GPU = 2             // 使用GPU完整预处理
};

// 后处理模式枚举
enum class PostprocessMode {
    CPU = 0,  // 使用CPU进行后处理（默认）
    GPU = 1   // 使用GPU进行后处理（CUDA加速）
};

// 帮助函数：从Flags获取全局预处理模式
inline PreprocessMode getGlobalPreprocessMode() {
    return static_cast<PreprocessMode>(FLAGS_global_preprocess_mode);
}

// 帮助函数：从Flags获取局部预处理模式
inline PreprocessMode getLocalPreprocessMode() {
    return static_cast<PreprocessMode>(FLAGS_local_preprocess_mode);
}

// 帮助函数：从Flags获取全局后处理模式
inline PostprocessMode getGlobalPostprocessMode() {
    return static_cast<PostprocessMode>(FLAGS_global_postprocess_mode);
        }

// 帮助函数：从Flags获取局部后处理模式
inline PostprocessMode getLocalPostprocessMode() {
    return static_cast<PostprocessMode>(FLAGS_local_postprocess_mode);
}

} // namespace tracking 