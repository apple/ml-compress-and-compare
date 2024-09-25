/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2024 Apple Inc. All Rights Reserved.
 */

import { Table, tableFromIPC } from "apache-arrow";
import pako from "pako";
import * as d3 from "d3";

export function readCompressedArrowBatch(
  compressed: string,
  base64 = true,
  gzipped = true
): Table {
  let intermediate: any = compressed;
  if (base64) intermediate = atob(intermediate);
  if (gzipped) {
    // Convert binary string to character-number array
    let charData = intermediate.split("").map((x: string) => x.charCodeAt(0));

    // Turn number array into byte-array
    let binData = new Uint8Array(charData);

    // Pako magic
    intermediate = pako.inflate(binData);
  }
  return tableFromIPC(intermediate);
}

export type Position = { x: number; y: number };
export type Rectangle = { x: number; y: number; width: number; height: number };

const positionHashMultiplier = Math.floor(Math.random() * 1e9);

export function hashPosition(pos: Position): number {
  return pos.x * positionHashMultiplier + pos.y;
}

export function positionsEqual(pos1: Position, pos2: Position): boolean {
  return pos1.x == pos2.x && pos1.y == pos2.y;
}

export function approxEquals(obj1: any, obj2: any): boolean {
  if (typeof obj1 == "number" && typeof obj2 == "number") {
    return Math.abs(obj1 - obj2) <= 0.001;
  }
  return obj1 == obj2;
}

export function makeTimeProvider() {
  let currentTime = 0;
  let fn = function () {
    return currentTime;
  };
  let advancer = Object.assign(fn, {
    advance(dt: number) {
      currentTime += dt;
    },
  });
  return advancer;
}

export let areSetsEqual = <T>(a: Set<T>, b: Set<T>): boolean =>
  a.size === b.size && [...a].every((value) => b.has(value));

export function cumulativeSum(arr: Array<number>): Array<number> {
  return arr.map(
    (
      (sum) => (value) =>
        (sum += value)
    )(0)
  );
}

export interface Histogram {
  [key: string]: number;
}

// https://stackoverflow.com/questions/1068834/object-comparison-in-javascript
export function objectsEqual(x: any, y: any) {
  if (x === y) return true;
  // if both x and y are null or undefined and exactly the same

  if (!(x instanceof Object) || !(y instanceof Object)) return false;
  // if they are not strictly equal, they both need to be Objects

  if (x.constructor !== y.constructor) return false;
  // they must have the exact same prototype chain, the closest we can do is
  // test there constructor.

  for (var p in x) {
    if (!x.hasOwnProperty(p)) continue;
    // other properties were tested using x.constructor === y.constructor

    if (!y.hasOwnProperty(p)) return false;
    // allows to compare x[ p ] and y[ p ] when set to undefined

    if (x[p] === y[p]) continue;
    // if they have the same strict value or identity then they are equal

    if (typeof x[p] !== "object") return false;
    // Numbers, Strings, Functions, Booleans must be strictly equal

    if (!objectsEqual(x[p], y[p])) return false;
    // Objects and Arrays must be tested recursively
  }

  for (p in y) if (y.hasOwnProperty(p) && !x.hasOwnProperty(p)) return false;
  // allows x[ p ] to be set to undefined

  return true;
}

export function wrapTextByCharacterCount(
  text: string,
  maxLineLength: number
): string {
  const words = text.replace(/[\r\n]+/g, " ").split(" ");
  let lineLength = 0;

  // use functional reduce, instead of for loop
  return words.reduce((result: string, word: string) => {
    if (lineLength + word.length >= maxLineLength) {
      lineLength = word.length;
      return result + `\n${word}`; // don't add spaces upfront
    } else {
      lineLength += word.length + (result ? 1 : 0);
      return result ? result + ` ${word}` : `${word}`; // add space only when needed
    }
  }, "");
}

/**
 * Uses objectsEqual to compute an array of the distinct objects
 * in the given array.
 */
export function distinctObjects<T>(objects: T[]): T[] {
  let positionValues: T[] = [];
  objects.forEach((val) => {
    if (positionValues.findIndex((v) => objectsEqual(v, val)) < 0)
      positionValues.push(val);
  });
  return positionValues;
}

/**
 * Uses canvas.measureText to compute and return the width of the given text of given font in pixels.
 *
 * @param {String} text The text to be rendered.
 * @param {String} font The css font descriptor that text is to be rendered with (e.g. "bold 14px verdana").
 *
 * @see https://stackoverflow.com/questions/118241/calculate-text-width-with-javascript/21015393#21015393
 */
export function getTextWidth(text: string, font: string) {
  // re-use canvas object for better performance
  var canvas: HTMLCanvasElement =
    getTextWidth.canvas ||
    (getTextWidth.canvas = document.createElement("canvas"));
  var context = canvas.getContext("2d");
  context.font = font;
  var metrics = context.measureText(text);
  return metrics.width;
}

export enum SortDirection {
  none = 0,
  descending = 1,
  ascending = 2,
}

export enum ComparisonSortDirection {
  none = 0,
  descending = 1,
  ascending = 2,
  differenceDescending = 3,
  differenceAscending = 4,
}

// Source https://stackoverflow.com/questions/16245767/creating-a-blob-from-a-base64-string-in-javascript
export function b64toBlob(
  b64Data: string,
  contentType: string = "image/png",
  sliceSize = 512
): Blob {
  const byteCharacters = atob(b64Data);
  const byteArrays = [];

  for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
    const slice = byteCharacters.slice(offset, offset + sliceSize);

    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }

  const blob = new Blob(byteArrays, { type: contentType });
  return blob;
}

export const schemeApple8 = [
  "#308BEF",
  "#4DC960",
  "#FF9500",
  "#EF5C53",
  "#A34FCD",
  "#00A39B",
  "#98685E",
  "#EEDB41",
];

export const SystemBlue = "#3378F6";
export const LightBlue = "#8cb4ff";
export const VeryLightBlue = "#e8f0ff";

export const SystemPink = "#EB4B63";

const _interpolateRed = d3.interpolateLab(SystemPink, "#e3e3e3");
const _interpolateBlue = d3.interpolateLab("#e3e3e3", SystemBlue);

export const interpolateSystemBlue = d3.interpolateHsl(
  VeryLightBlue,
  SystemBlue
);

export const interpolateSystemRedBlue: (t: number) => string = function (t) {
  if (t <= 0.5) return _interpolateRed(t * 2);
  return _interpolateBlue(t * 2 - 1);
};

/**
 * Calculate brightness value by RGB or HEX color.
 * @param color (String) The color value in RGB or HEX (for example: #000000 || #000 || rgb(0,0,0) || rgba(0,0,0,0))
 * @returns (Number) The brightness value (dark) 0 ... 255 (light)
 */
export function brightnessByColor(colorObj: any): number {
  let color = "" + colorObj,
    isHEX = color.indexOf("#") == 0,
    isRGB = color.indexOf("rgb") == 0;
  let r: number | undefined, g: number | undefined, b: number | undefined;
  if (isHEX) {
    const hasFullSpec = color.length == 7;
    let m = color.substr(1).match(hasFullSpec ? /(\S{2})/g : /(\S{1})/g);
    if (m)
      (r = parseInt(m[0] + (hasFullSpec ? "" : m[0]), 16)),
        (g = parseInt(m[1] + (hasFullSpec ? "" : m[1]), 16)),
        (b = parseInt(m[2] + (hasFullSpec ? "" : m[2]), 16));
  }
  if (isRGB) {
    let m = color.match(/(\d+){3}/g);
    if (m) (r = parseInt(m[0])), (g = parseInt(m[1])), (b = parseInt(m[2]));
  }
  if (r !== undefined && g !== undefined && b !== undefined)
    return (r * 299 + g * 587 + b * 114) / 1000;
  return 0;
}
