class Greeter {
  constructor(name) {
    this.name = name;
  }
  greet(greeting) {
    return greeting + this.name;
  }
  static create(name) {
    return new Greeter(name);
  }
}

function standalone(x) {
  return x * 2;
}
