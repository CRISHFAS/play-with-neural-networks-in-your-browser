import {Example2D} from "./dataset";
import d3 from 'd3';
export interface HeatMapSettings {
  [key: string]: any;
  showAxes?: boolean;
  noSvg?: boolean;
}

const NUM_SHADES = 30;

export class HeatMap {
  private settings: HeatMapSettings = {
    showAxes: false,
    noSvg: false
  };
  private xScale;
  private yScale;
  private numSamples: number;
  private color;
  private canvas;
  private svg;

  constructor(
      width: number, numSamples: number, xDomain: [number, number],
      yDomain: [number, number], container,
      userSettings?: HeatMapSettings) {
    this.numSamples = numSamples;
    let height = width;
    let padding = userSettings.showAxes ? 20 : 0;

    if (userSettings != null) {
      for (let prop in userSettings) {
        this.settings[prop] = userSettings[prop];
      }
    }

    this.xScale = d3.scale.linear()
      .domain(xDomain)
      .range([0, width - 2 * padding]);

    this.yScale = d3.scale.linear()
      .domain(yDomain)
      .range([height - 2 * padding, 0]);

    let tmpScale = d3.scale.linear<string, number>()
        .domain([0, .5, 1])
        .range(["#f59322", "#e8eaeb", "#0877bd"])
        .clamp(true);

    let colors = d3.range(0, 1 + 1E-9, 1 / NUM_SHADES).map(a => {
      return tmpScale(a);
    });
    this.color = d3.scale.quantize()
                     .domain([-1, 1])
                     .range(colors);

    container = container.append("div")
      .style({
        width: `${width}px`,
        height: `${height}px`,
        position: "relative",
        top: `-${padding}px`,
        left: `-${padding}px`
      });
    this.canvas = container.append("canvas")
      .attr("width", numSamples)
      .attr("height", numSamples)
      .style("width", (width - 2 * padding) + "px")
      .style("height", (height - 2 * padding) + "px")
      .style("position", "absolute")
      .style("top", `${padding}px`)
      .style("left", `${padding}px`);

    if (!this.settings.noSvg) {
      this.svg = container.append("svg").attr({
          "width": width,
          "height": height
      }).style({
        "position": "absolute",
        "left": "0",
        "top": "0"
      }).append("g")
        .attr("transform", `translate(${padding},${padding})`);

      this.svg.append("g").attr("class", "train");
      this.svg.append("g").attr("class", "test");
    }

    if (this.settings.showAxes) {
      let xAxis = d3.svg.axis()
        .scale(this.xScale)
        .orient("bottom");

      let yAxis = d3.svg.axis()
        .scale(this.yScale)
        .orient("right");

      this.svg.append("g")
        .attr("class", "eje x")
        .attr("transform", `translate(0,${height - 2 * padding})`)
        .call(xAxis);

      this.svg.append("g")
        .attr("class", "eje y")
        .attr("transform", "translate(" + (width - 2 * padding) + ",0)")
        .call(yAxis);
    }
  }

  updateTestPoints(points: Example2D[]): void {
    if (this.settings.noSvg) {
      throw Error("No se pueden agregar puntos porque noSvg=true");
    }
    this.updateCircles(this.svg.select("g.test"), points);
  }

  updatePoints(points: Example2D[]): void {
    if (this.settings.noSvg) {
      throw Error("No se pueden agregar puntos porque noSvg=true");
    }
    this.updateCircles(this.svg.select("g.train"), points);
  }

  updateBackground(data: number[][], discretize: boolean): void {
    let dx = data[0].length;
    let dy = data.length;

    if (dx !== this.numSamples || dy !== this.numSamples) {
      throw new Error(
          "La matriz de datos proporcionada debe tener un tamaño " +
          "numMuestras X numMuestras");
    }

    let context = (this.canvas.node() as HTMLCanvasElement).getContext("2d");
    let image = context.createImageData(dx, dy);

    for (let y = 0, p = -1; y < dy; ++y) {
      for (let x = 0; x < dx; ++x) {
        let value = data[x][y];
        if (discretize) {
          value = (value >= 0 ? 1 : -1);
        }
        let c = d3.rgb(this.color(value));
        image.data[++p] = c.r;
        image.data[++p] = c.g;
        image.data[++p] = c.b;
        image.data[++p] = 160;
      }
    }
    context.putImageData(image, 0, 0);
  }

  private updateCircles(container, points: Example2D[]) {
    let xDomain = this.xScale.domain();
    let yDomain = this.yScale.domain();
    points = points.filter(p => {
      return p.x >= xDomain[0] && p.x <= xDomain[1]
        && p.y >= yDomain[0] && p.y <= yDomain[1];
    });

    let selection = container.selectAll("circle").data(points);

    selection.enter().append("circle").attr("r", 3);

    selection
      .attr({
        cx: (d: Example2D) => this.xScale(d.x),
        cy: (d: Example2D) => this.yScale(d.y),
      })
      .style("fill", d => this.color(d.label));

    selection.exit().remove();
  }
}  

export function reduceMatrix(matrix: number[][], factor: number): number[][] {
  if (matrix.length !== matrix[0].length) {
    throw new Error("La matriz proporcionada debe ser una matriz cuadrada");
  }
  if (matrix.length % factor !== 0) {
    throw new Error("El ancho/alto de la matriz debe ser divisible por " +
        "el factor de reducción");
  }
  let result: number[][] = new Array(matrix.length / factor);
  for (let i = 0; i < matrix.length; i += factor) {
    result[i / factor] = new Array(matrix.length / factor);
    for (let j = 0; j < matrix.length; j += factor) {
      let avg = 0;
      for (let k = 0; k < factor; k++) {
        for (let l = 0; l < factor; l++) {
          avg += matrix[i + k][j + l];
        }
      }
      avg /= (factor * factor);
      result[i / factor][j / factor] = avg;
    }
  }
  return result;
}