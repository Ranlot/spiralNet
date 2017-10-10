#!/usr/bin/env bash

numbClasses=$1

function movieMaker {
  echo $1
  convert -loop 0 -delay 10 $(ls $1 | sort -n -t. -k2) $2
}

movieMaker "framesDir/$numbClasses/decisionBoundaries.*.png" "framesDir/hidden.anim.$numbClasses.gif"
