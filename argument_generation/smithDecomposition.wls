#!/usr/bin/env wolframscript
(* ::Package:: *)

args = $ScriptCommandLine[[2;;]];
matrix = ToExpression[args[[1]]];
outputFile = args[[2]];

decomposition = SmithDecomposition[matrix];

Export[outputFile, decomposition, "Text"];
