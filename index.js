const set = [
  {
    inputs: [32, 175],
    expected: 1
  },
  {
    inputs: [24, 170],
    expected: 1
  },
  {
    inputs: [20, 50],
    expected: 1
  },
  {
    inputs: [30, 10],
    expected: 0
  },
  {
    inputs: [24, 340],
    expected: 1
  },
  {
    inputs: [64, 250],
    expected: 0
  },
  {
    inputs: [34, 120],
    expected: 0
  }
];
const max = set.reduce((result, { inputs }) => inputs.map((input, key) => result[key] > input ? result[key] : input), []);
const trainingSet = set.map(({ inputs, expected }) => ({
  inputs: inputs.map((input, key) => input/max[key]),
  expected
}));
const weights = length => Array(length).fill(Math.random());
const weightedSum = weights => inputs => inputs.reduce((result, input, key) => input * weights[key] + result, 0);
const activate = value => value >= 0 ? 1 : 0;
const delta = learningRate => actual => expected => input => expected - actual * learningRate * input;
const countWeights = weights => trainingSet => trainingSet.reduce((result, { inputs, expected }) => inputs.map((input, key) => weights[key] + delta(.1)(activate(weightedSum(weights)(inputs)))(expected)(input)), []);
const learn = weights => trainingSet => trainingSet.every(({ inputs, expected }) => activate(weightedSum(weights)(inputs)) === expected) ? weights : countWeights(weights)(trainingSet)//learn(countWeights(weights)(trainingSet))(trainingSet);

console.log(learn(weights(max.length))(trainingSet));
