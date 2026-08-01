package main

import "fmt"

type S struct {
	X int
}

func NewS() *S {
	return &S{}
}

func (s *S) M() {
	s.X++
	fmt.Println(s.X)
}
