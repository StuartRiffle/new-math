@echo off
setlocal enabledelayedexpansion

pushd %~dp0

copy /y automaton.py _code.txt

for %%1 in (small large huge) do (
    del notes-%%1.txt
    for /f "delims=" %%f in ('dir /b /on glom\*.txt') do type glom\%%f >> notes-%%1.txt
    for /f "delims=" %%f in ('dir /b /on glom\%%1\*.txt') do type glom\%%1\%%f >> notes-%%1.txt
    type _code.txt >> notes-%%1.txt
    for /f "delims=" %%f in ('dir /b /on glom\final\*.txt') do type glom\final\%%f >> notes-%%1.txt
)

del _code.txt
