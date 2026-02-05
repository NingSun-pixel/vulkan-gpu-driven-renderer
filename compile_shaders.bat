@echo off
setlocal enabledelayedexpansion

:: ==========================================
:: 自动编译 Shaders 脚本
:: ==========================================

:: 1. 切换到脚本所在的目录（保证路径正确）
cd /d "%~dp0"

:: 2. 检查 glslc 命令是否存在 (Vulkan SDK 是否安装并配置环境变量)
where glslc >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 找不到 glslc.exe ！
    echo 请确认你已经安装了 Vulkan SDK，并且将 Bin 目录加到了系统环境变量 PATH 中。
    pause
    exit /b
)

:: 3. 进入 shaders 文件夹
if not exist "shaders" (
    echo [ERROR] 当前目录下找不到 "shaders" 文件夹！
    echo 请把这个脚本放在项目根目录，也就是 shaders 文件夹的上一级。
    pause
    exit /b
)

cd shaders
echo [INFO] Entering shaders directory: %cd%
echo ----------------------------------------------

:: 4. 遍历所有常见的 Shader 后缀并编译
:: %%f 是文件名 (gradient.comp)
:: -o 后面指定输出文件名 (gradient.comp.spv)

set found=0
for %%f in (*.vert *.frag *.comp *.geom *.tesc *.tese) do (
    set found=1
    echo [Compiling] %%f ...
    
    :: 执行编译命令
    glslc "%%f" -o "%%f.spv"
    
    :: 检查上一条命令是否成功
    if !errorlevel! neq 0 (
        echo [FAILED] 编译 %%f 失败！请检查代码错误。
        color 4F
    ) else (
        echo    -^> Generated: %%f.spv
    )
    echo.
)

if %found%==0 (
    echo [WARNING] 没有找到任何 shader 文件。
)

echo ----------------------------------------------
echo [INFO] Compilation finished.
pause