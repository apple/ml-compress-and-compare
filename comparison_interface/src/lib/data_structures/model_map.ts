/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2024 Apple Inc. All Rights Reserved.
 */

export interface OperationParameterSpec {
  type: string; // continuous, nominal, ordinal
  min?: number;
  max?: number;
  options?: string[]; // for nominal or ordinal parameters
  format?: string; // d3 format string for numerical parameters
}

export interface CompressionOperationSpec {
  name: string;
  type?: string; // currently unused
  parameters: { [key: string]: OperationParameterSpec };
}

export interface CompressionOperation {
  name: string;
  parameters: { [key: string]: any };
}

export function operationToString(
  operation: CompressionOperation,
  full: boolean = true
): string {
  return (
    `${operation.name} ` +
    Object.entries(operation.parameters)
      .map(([name, val]) => (full ? `${name} = ${val}` : `${val}`))
      .join(", ")
  );
}

export interface Metric {
  name: string;
  type?: string; // currently unused
  primary?: boolean;
  format?: string;
  unit?: string;
  min?: number;
  max?: number;
}

export interface Model {
  id: string;
  tag?: string;
  base?: string;
  operation?: CompressionOperation;
  metrics: { [key: string]: any };
}

export function calculateDomain(
  metrics: Metric[],
  models: Model[],
  metricName: string
): [number, number] {
  if (!metricName) return [0, 1];
  let metricInfo = metrics.find((m) => m.name == metricName);
  let metricMin: number | null = null;
  let metricMax: number | null = null;
  if (!!metricInfo && metricInfo.min !== undefined) metricMin = metricInfo.min;
  if (!!metricInfo && metricInfo.max !== undefined) metricMax = metricInfo.max;
  let metricValues = models.map((m) => m.metrics[metricName]);
  if (metricMin == null)
    metricMin = metricValues.reduce((prev, curr) => Math.min(prev, curr), 1e9);
  if (metricMax == null)
    metricMax = metricValues.reduce((prev, curr) => Math.max(prev, curr), -1e9);
  console.log(metricName, metricMin, metricMax);
  return [metricMin!, metricMax!];
}
