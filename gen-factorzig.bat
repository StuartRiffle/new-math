setlocal enabledelayedexpansion
pushd %~dp0

canon.py table --cols "~factorzig" --max 1009 --factor-exponents --factor-dots --csv --pad > data\raw-factorzig-1000.txt
textfix --columns --column-length 98 --column-margin 3 --column-split-regex "^^\d" data\raw-factorzig-1000.txt > data\factorzig-1000-cols.txt

