@echo off
setlocal enabledelayedexpansion
set DEST=shape-of-integers.tex.txt
textfix --preprocess draft.tex.txt > %dest%
call toks %dest%

