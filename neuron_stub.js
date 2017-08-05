class Neuron {
	/**
	 * @param {Number} bias - вес вывода (в случае одного нейрона всегда 1)
	 * @param {Number} learningRate - шаг обучения (Слишком большой шаг - высокая вероятность "проскочить" нужное значение, слишком малый - долгое обучение)
	 * @param {Array} weights - инициализация "веса" для каждого ввода нейрона
	 *
	 * @public
	 */
	constructor(bias=1, learningRate=0.1, weights=[]) {
		this.bias = bias;
		this.learningRate = learningRate;
		this.weights = weights;
		this.trainingSet = [];
	}

	/**
	 * Метод активации
	 * @param {Number} value - значение weightedSum
	 *
	 * @returns {Number}
	 *
	 * @public
	 */
	activate(value) {
		return value >= 0 ? 1 : 0;
	}

	/**
	 * Метод расчета суммы произведений входящих значений на веса
	 * @param {Array} inputs - ввод
	 * @param {Array} weights - веса
	 * @returns {Number}
	 *
	 * @public
	 */
	weightedSum(inputs=this.inputs, weights=this.weights) {
		return inputs.map((inp, i) => inp * weights[i]).reduce((x, y) => x + y, 0);
	}

	/**
	 * Метод активации
	 * @param {Array} inputs - ввод
	 * @returns {Number}
	 *
	 * @public
	 */
	evaluate(inputs) {
		return this.activate(this.weightedSum(inputs));
	}

	/**
	 * Метод инициализации (проставление рандомных весов)
	 * @param {Array} inputs - ввод
	 * @param {Number} bias - вес вывода (в случае одного нейрона всегда 1)
	 * @returns {Number}
	 *
	 * @public
	 */
	init(inputs, bias=this.bias) {
		this.weights = [...inputs.map(i => Math.random()), bias];
	}

	/**
	 * Метод вычисления величины изменения каждого веса в случае ошибки
	 * @param {Number} actual - значение вывода нейрона на текущей выборке
	 * @param {Number} expected - ожидаемое значение вывода
	 * @param {Array} input - значение ввода на текущем шаге
	 * @param {Number} learningRate - шаг обучения
	 * @returns {Number} - величина изменения веса для текущего ввода
	 *
	 * @public
	 */
	delta(actual, expected, input, learningRate=this.learningRate) {
		const error = expected - actual;

		return error * learningRate * input;
	}

	/**
	 * Метод тренировки нейрона на одной обучающей выборке
	 * @param {Array} inputs - входящая выборка
	 * @param {Number} expected - ожидаемое значение вывода
	 *
	 * @returns {Array|Boolean} - Измененные веса, либо true если вывод совпал с ожиданием
	 *
	 */
	train(inputs, expected) {
		if (!this.weights.length) this.init(inputs);
		if (inputs.length != this.weights.length) inputs.push(1); // Adding the bias

		// Keeping this in the training set if it didn't exist
		if (!this.trainingSet.find(t => t.inputs.every((inp,i) => inp === inputs[i]))) this.trainingSet.push({inputs,expected});

		const actual = this.evaluate(inputs);
		if (actual == expected) return true; // Correct weights return and don't touch anything.

		// Otherwise update each weight by adding the error * learningRate relative to the input
		this.weights = this.weights.map((w,i) => w += this.delta(actual, expected,inputs[i]));
		return this.weights;
	}

	/**
	 * Метод обучения на обучающем наборе
	 * @param {Function} iterationCallback - обратный вызов для каждой итерации
	 * @param {Array} trainingSet - обучающий набор
	 *
	 */
	learn(iterationCallback=()=>{}, trainingSet=this.trainingSet) {
		let success = false;
		while (!success) {
			// Function of your choosing that will be called after an iteration has completed
			iterationCallback.call(this);
			success = trainingSet.every(t => this.train(t.inputs,t.expected) === true);
		}
	}

	predict(inputs, bias) {
		return this.evaluate([...inputs, bias]);
	}
}

module.exports = Neuron;
