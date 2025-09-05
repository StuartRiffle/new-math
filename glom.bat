@echo off
setlocal enabledelayedexpansion

pushd %~dp0

copy /y automaton.py glom\78-automaton.txt

for %%1 in (small large huge) do (
    del notes-%%1-*kt.txt >NUL 2>NUL
    for /f "delims=" %%f in ('dir /b /on glom\*.txt') do type glom\%%f >> notes-%%1-0kt.txt
    for /f "delims=" %%f in ('dir /b /on glom\%%1\*.txt') do type glom\%%1\%%f >> notes-%%1-0kt.txt
    for /f "delims=" %%f in ('dir /b /on glom\final\*.txt') do type glom\final\%%f >> notes-%%1-0kt.txt
)

call toks --update notes-*-*kt.txt
