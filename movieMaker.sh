#!/usr/bin/env bash

function movieMaker {
  echo $1
  convert -loop 0 -delay 10 $(ls $1 | sort -n -t. -k2) $2
}

movieMaker "framesDir/decisionBoundaries.*.png" "hidden.anim.gif"
