@echo off
setlocal enabledelayedexpansion

pushd %~dp0

for %%1 in (small large huge) do (
    del notes-%%1.txt
    for /f "delims=" %%f in ('dir /b /on glom\*.txt') do type glom\%%f >> notes-%%1.txt
    for /f "delims=" %%f in ('dir /b /on glom\%%1\*.txt') do type glom\%%1\%%f >> notes-%%1.txt
)
