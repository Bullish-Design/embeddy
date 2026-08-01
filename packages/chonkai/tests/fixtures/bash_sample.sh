#!/bin/bash

greet() {
    echo "hello $1"
}

function add() {
    local sum=$(( $1 + $2 ))
    echo "$sum"
}

greet "world"
add 1 2
