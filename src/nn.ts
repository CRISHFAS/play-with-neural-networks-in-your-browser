/**
* Un nodo en una red neuronal. Cada nodo tiene un estado
* (entrada total, salida y sus respectivas derivadas) que cambia
* después de cada ejecución de propagación hacia adelante y hacia atrás.
*/
export class Node {
  id: string;
  /** Lista de enlaces de entrada. */
  inputLinks: Link[] = [];
  bias = 0.1;
  /** Lista de enlaces de salida. */
  outputs: Link[] = [];
  totalInput: number;
  output: number;
  /** Error derivado con respecto a la salida de este nodo. */
  outputDer = 0;
  /** Derivada del error con respecto a la entrada total de este nodo. */
  inputDer = 0;
  /**
  * Derivada del error acumulado con respecto a la entrada total de este nodo desde la última actualización. Esta derivada es igual a dE/db, donde b es el término de sesgo del nodo.
  */
  accInputDer = 0;
  /**
  * Número de derivadas de error acumuladas con respecto a la entrada total
  * desde la última actualización.
  */
  numAccumulatedDers = 0;
  /** Función de activación que toma la entrada total y devuelve la salida del nodo */
  activation: ActivationFunction;

  /**
  * Crea un nuevo nodo con el ID y la función de activación proporcionados.
  */
  constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
    this.id = id;
    this.activation = activation;
    if (initZero) {
      this.bias = 0;
    }
  }

  /** Recalcula la salida del nodo y la devuelve. */
  updateOutput(): number {
    // Almacena la entrada total en el nodo.
    this.totalInput = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.totalInput += link.weight * link.source.output;
    }
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }
}

/**
* Una función de error y su derivada.
*/
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** Función de activación de un nodo y su derivada. */
export interface ActivationFunction {
  output: (input: number) => number;
  der: (input: number) => number;
}

/** Función que calcula un costo de penalización para un peso dado en la red. */
export interface RegularizationFunction {
  output: (weight: number) => number;
  der: (weight: number) => number;
}

/** Funciones de error integradas */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
               0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target
  };
}

/** Polyfill para TANH */
(Math as any).tanh = (Math as any).tanh || function(x) {
  if (x === Infinity) {
    return 1;
  } else if (x === -Infinity) {
    return -1;
  } else {
    let e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
};

/** Funciones de activación integradas */
export class Activations {
  public static TANH: ActivationFunction = {
    output: x => (Math as any).tanh(x),
    der: x => {
      let output = Activations.TANH.output(x);
      return 1 - output * output;
    }
  };
  public static RELU: ActivationFunction = {
    output: x => Math.max(0, x),
    der: x => x <= 0 ? 0 : 1
  };
  public static SIGMOID: ActivationFunction = {
    output: x => 1 / (1 + Math.exp(-x)),
    der: x => {
      let output = Activations.SIGMOID.output(x);
      return output * (1 - output);
    }
  };
  public static LINEAR: ActivationFunction = {
    output: x => x,
    der: x => 1
  };
}

/** Funciones de regularización incorporadas */
export class RegularizationFunction {
  public static L1: RegularizationFunction = {
    output: w => Math.abs(w),
    der: w => w < 0 ? -1 : (w > 0 ? 1 : 0)
  };
  public static L2: RegularizationFunction = {
    output: w => 0.5 * w * w,
    der: w => w
  };
}

/**
* Un enlace en una red neuronal. Cada enlace tiene un peso y un nodo de origen y
* de destino. También tiene un estado interno (derivada del error
* con respecto a una entrada específica) que se actualiza después
* de una serie de retropropagación.
*/
export class Link {
  id: string;
  source: Node;
  dest: Node;
  weight = Math.random() - 0.5;
  isDead = false;
  /** Error derivado con respecto a este peso. */
  errorDer = 0;
  /** Accumulated error derivative since the last update. */
  accErrorDer = 0;
  /** Error acumulado derivado desde la última actualización. */
  numAccumulatedDers = 0;
  regularization: RegularizationFunction;

  /**
   * Construye un enlace en la red neuronal inicializada con peso aleatorio.
   *
   * @param source El nodo de origen.
   * @param dest El nodo de destino.
   * @param regularization La función de regularización que calcula la
   *     penalización por este peso. Si es nulo, no habrá regularización..
   */
  constructor(source: Node, dest: Node,
      regularization: RegularizationFunction, initZero?: boolean) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    if (initZero) {
      this.weight = 0;
    }
  }
}

/**
 * Construye una red neuronal.
 *
 * @param networkShape La estructura de la red. Por ejemplo, [1, 2, 3, 1] significa
 *   que la red tendrá un nodo de entrada, 2 nodos en la primera capa oculta,
 *   3 nodos en la segunda capa oculta y 1 nodo de salida.
 * @param activation La función de activación de cada nodo oculto.
 * @param outputActivation La función de activación para los nodos de salida.
 * @param regularization La función de regularización que calcula una penalización
 *     para un peso (parámetro) dado en la red. Si es null, no habrá
 *     regularización.
 * @param inputIds Lista de identificadores para los nodos de entrada.
 */

export function buildNetwork(
    networkShape: number[], activation: ActivationFunction,
    outputActivation: ActivationFunction,
    regularization: RegularizationFunction,
    inputIds: string[], initZero?: boolean): Node[][] {
  let numLayers = networkShape.length;
  let id = 1;
  /** Lista de capas, donde cada capa es una lista de nodos. */
  let network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;
    let isInputLayer = layerIdx === 0;
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      let node = new Node(nodeId,
          isOutputLayer ? outputActivation : activation, initZero);
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Agrega enlaces desde los nodos de la capa anterior a este nodo.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node, regularization, initZero);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }
  return network;
}

/**
 * Ejecuta una propagación hacia adelante de la entrada proporcionada a través de la red
 * proporcionada. Este método modifica el estado interno de la red: la
 * entrada total y la salida de cada nodo en la red.
 *
 * @param network La red neuronal.
 * @param inputs El arreglo de entrada. Su longitud debe coincidir con el número de nodos
 *     de entrada en la red.
 * @return La salida final de la red.
 */

export function forwardProp(network: Node[][], inputs: number[]): number {
  let inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
        " the input layer");
  }
  // Actualizar la capa de entrada.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    node.output = inputs[i];
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Actualizar todos los nodos en esta capa.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.updateOutput();
    }
  }
  return network[network.length - 1][0].output;
}

/**
* Ejecuta una propagación hacia atrás utilizando el objetivo proporcionado y la
* salida calculada de la llamada anterior a la propagación hacia adelante.
* Este método modifica el estado interno de la red: las derivadas de error
* con respecto a cada nodo y cada peso
* en la red.
*/
export function backProp(network: Node[][], target: number,
    errorFunc: ErrorFunction): void {
  // El nodo de salida es un caso especial. Usamos el error definido por el usuario.
  // función para la derivada.
  let outputNode = network[network.length - 1][0];
  outputNode.outputDer = errorFunc.der(outputNode.output, target);

  // Vaya a través de las capas hacia atrás.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx];
    // Calcular la derivada del error de cada nodo con respecto a:
    // 1) su entrada total
    // 2) cada uno de sus pesos de entrada.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.inputDer = node.outputDer * node.activation.der(node.totalInput);
      node.accInputDer += node.inputDer;
      node.numAccumulatedDers++;
    }

    // Derivada del error con respecto a cada peso que entra al nodo.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        link.errorDer = node.inputDer * link.source.output;
        link.accErrorDer += link.errorDer;
        link.numAccumulatedDers++;
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    let prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      let node = prevLayer[i];
      // Calcula la derivada del error con respecto a la salida de cada nodo.
      node.outputDer = 0;
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        node.outputDer += output.weight * output.dest.inputDer;
      }
    }
  }
}

/**
* Actualiza los pesos de la red utilizando las derivadas del error acumulado previamente.
*/
export function updateWeights(network: Node[][], learningRate: number,
    regularizationRate: number) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // Actualiza el sesgo del nodo.
      if (node.numAccumulatedDers > 0) {
        node.bias -= learningRate * node.accInputDer / node.numAccumulatedDers;
        node.accInputDer = 0;
        node.numAccumulatedDers = 0;
      }
      // Actualiza los pesos que llegan a este nodo.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        let regulDer = link.regularization ?
            link.regularization.der(link.weight) : 0;
        if (link.numAccumulatedDers > 0) {
          // Actualizar el peso en función de dE/dw.
          link.weight = link.weight -
              (learningRate / link.numAccumulatedDers) * link.accErrorDer;
          // Actualizar aún más el peso en función de la regularización.
          let newLinkWeight = link.weight -
              (learningRate * regularizationRate) * regulDer;
          if (link.regularization === RegularizationFunction.L1 &&
              link.weight * newLinkWeight < 0) {
            // El peso superó el valor 0 debido al término de regularización. Establézcalo en 0.
            link.weight = 0;
            link.isDead = true;
          } else {
            link.weight = newLinkWeight;
          }
          link.accErrorDer = 0;
          link.numAccumulatedDers = 0;
        }
      }
    }
  }
}

/** Itera sobre cada nodo de la red/ */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
    accessor: (node: Node) => any) {
  for (let layerIdx = ignoreInputs ? 1 : 0;
      layerIdx < network.length;
      layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      accessor(node);
    }
  }
}

/** Devuelve el nodo de salida en la red. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}