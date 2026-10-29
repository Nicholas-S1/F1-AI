@echo off
setlocal
cd /d "%~dp0"
set REPO_URL=__REPO_URL__

echo === F1 AI environment setup ===

rem --- find Python ---
where py >nul 2>&1 && (set PY=py -3) || (
  where python >nul 2>&1 && (set PY=python) || (
    echo [!] Python not found. Install it from https://www.python.org/downloads/
    echo     and tick "Add python.exe to PATH", then run setup.bat again.
    pause & exit /b 1
  )
)
%PY% --version

rem --- virtual environment + packages ---
if not exist .venv (
  echo Creating .venv ...
  %PY% -m venv .venv || (echo [!] venv creation failed & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt || (echo [!] pip install failed & pause & exit /b 1)
python -c "import pygame; print('pygame', pygame.version.ver, 'OK')"

rem --- VS Code settings: use the venv interpreter ---
if not exist .vscode mkdir .vscode
if not exist .vscode\settings.json (
  > .vscode\settings.json (
    echo {
    echo     "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
    echo     "python.terminal.activateEnvironment": true
    echo }
  )
)
if not exist .vscode\extensions.json (
  > .vscode\extensions.json (
    echo { "recommendations": ["ms-python.python", "ms-python.debugpy"] }
  )
)

rem --- git: first-time init and push (skipped when cloned) ---
where git >nul 2>&1 || (
  echo [!] Git not found. Install from https://git-scm.com/download/win to use GitHub.
  goto done
)
if not exist .git (
  git init -b main
  git add .
  git commit -m "Initial project setup: venv, pygame, VS Code settings"
  git remote add origin %REPO_URL%
  git push -u origin main
)

:done
echo.
echo === Setup complete. Open this folder in VS Code. ===
pause
