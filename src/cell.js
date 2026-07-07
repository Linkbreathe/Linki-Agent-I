class Cell {
  constructor(alive) {
    this._alive = !!alive;
  }

  isAlive() {
    return this._alive;
  }

  toString() {
    return this._alive ? '+' : '.';
  }
}

function createAlive() {
  return new Cell(true);
}

function createDead() {
  return new Cell(false);
}

function fromBoolean(value) {
  return new Cell(value);
}

module.exports = { Cell, createAlive, createDead, fromBoolean };
