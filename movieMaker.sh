#!/usr/bin/env bash

numbClasses=$1
activation=$2

function movieMaker {
  echo $1
  convert -loop 0 -delay 10 $(ls $1 | sort -n -t. -k3) $2
}

movieMaker "framesDir/$numbClasses/$activation.decisionBoundaries.*.png" "plotDir/$numbClasses/$activation.hidden.anim.gif"
