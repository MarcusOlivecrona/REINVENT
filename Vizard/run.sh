#!/bin/bash
if [ -z "$1" ];
    then echo "Must supply path to a directory where vizard_logger is saving its information";
    exit 0
fi
bokeh serve . --args $1
