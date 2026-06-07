@echo off
REM ===========================================================================
REM TaxTronik / Risk-Layer - Alltagsstart (Doppelklick). KEIN Re-Setup.
REM Startet die /v1-Engine (:8000) UND Schicht 2 (llama-server :8080) in EINEM
REM Befehl. Erstmaliges Setup vorher einmal mit deploy.ps1.
REM Mandantengeheimnis (Par.203): bindet nur auf 127.0.0.1, nichts verlaesst den PC.
REM ===========================================================================
cd /d "%~dp0"

REM Token (fuer Loopback nicht noetig, aber harmlos vorzuladen)
if exist ".engine_token" (
  for /f "usebackq delims=" %%t in (".engine_token") do set "RISK_LAYER_TOKEN=%%t"
)
if not defined RISK_LAYER_EMB_DEVICE set "RISK_LAYER_EMB_DEVICE=cuda"

echo Starte Risk-Layer  ( /v1 :8000  +  LLM :8080 ) ...
echo Strg+C beendet beides.
python -m risk_layer.web --host 127.0.0.1 --port 8000 --llm-autostart

echo.
echo Engine beendet.
pause
