REM @echo off
setlocal enabledelayedexpansion

pushd %~dp0

canon.py table --cols "~factorzig" --max 1050 --factor-exponents --factor-dots --csv --pad > data\raw-factorzig-1000.txt
