/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2024 Apple Inc. All Rights Reserved.
 */

export interface Prediction {
  pred: string | number;
}

export interface InstanceData {
  text?: string;
  image?: string;
  imageFormat?: string;
}

type ComparisonSet = {
  [key: string]: { [key: string]: { difference: number; value?: number } };
};
export interface PredictionContainerBase {
  comparisons: ComparisonSet;
}

export interface InstancePrediction extends PredictionContainerBase {
  id: any;
  label: string; // ground-truth label
  classes?: string[]; // potentially multiple labels that can be filtered for
  predictions: { [key: string]: Prediction };
}

export interface ClassPrediction extends PredictionContainerBase {
  class: string;
  predictions: { [key: string]: Prediction[] };
}

function emptyComparisonObjectLike(comparisons: ComparisonSet): ComparisonSet {
  return Object.fromEntries(
    Object.keys(comparisons).map((compName) => [
      compName,
      Object.fromEntries(
        Object.keys(comparisons[compName]).map((m) => [
          m,
          {
            value: comparisons[compName][m].value != undefined ? 0 : undefined,
            difference: 0,
          },
        ])
      ),
    ])
  );
}
/**
 * A class that manages a list of predictions (either instance-level or class-level)
 * and enables filtering, sorting, and pagination on multiple comparison metrics
 * stored for each prediction.
 */
export class PredictionDataset {
  predictions: PredictionContainerBase[];
  modelNames: string[];
  length: number;
  isClassLevel: boolean;

  constructor(
    predictions: PredictionContainerBase[],
    modelNames: string[],
    isClassLevel: boolean | null = null
  ) {
    this.predictions = predictions;
    this.length = predictions.length;
    this.modelNames = modelNames;
    if (isClassLevel === null)
      this.isClassLevel =
        predictions.length > 0 && predictions[0].hasOwnProperty("class");
    else this.isClassLevel = isClassLevel;
  }

  hasRawValues(comparisonName: string): boolean {
    return (
      Object.values(this.predictions[0].comparisons[comparisonName])[0].value !=
      undefined
    );
  }

  filterPredictions(
    filterFn: (prediction: PredictionContainerBase) => boolean
  ): PredictionDataset {
    return new PredictionDataset(
      this.predictions.filter(filterFn),
      this.modelNames,
      this.isClassLevel
    );
  }

  filterByComparisonMetric(
    comparisonName: string,
    rawValue: boolean,
    filterFn: (val: number, modelID: string, index: number) => boolean
  ): PredictionDataset {
    const scoreKey = rawValue ? "value" : "difference";
    if (this.predictions.length > 0 && this.modelNames.length > 0)
      console.log(
        this.predictions[0],
        this.modelNames,
        comparisonName,
        this.predictions[0].comparisons[comparisonName]
      );
    if (
      this.predictions.length > 0 &&
      this.modelNames.length > 0 &&
      !this.predictions[0].comparisons[comparisonName][
        this.modelNames[0]
      ].hasOwnProperty(scoreKey)
    ) {
      console.error(`Can't get key '${scoreKey}', missing from predictions`);
      return this;
    }
    return new PredictionDataset(
      this.predictions.filter((p) => {
        return this.modelNames.every((modelID, i) =>
          filterFn(
            p.comparisons[comparisonName][modelID][scoreKey]!,
            modelID,
            i
          )
        );
      }),
      this.modelNames,
      this.isClassLevel
    );
  }

  getClasses(): string[] {
    if (this.isClassLevel) {
      return Array.from(
        new Set(this.predictions.map((p) => (p as ClassPrediction).class))
      ).sort();
    }
    return Array.from(
      new Set(
        this.predictions
          .map((p) => (p as InstancePrediction).classes || [])
          .flat()
      )
    ).sort();
  }

  getIDs(): any[] {
    if (this.isClassLevel)
      return this.predictions.map((p) => (p as ClassPrediction).class);
    return this.predictions.map((p) => (p as InstancePrediction).id);
  }

  sortByID(): PredictionDataset {
    let newPredictions = Array.from(this.predictions);
    if (this.isClassLevel)
      newPredictions.sort((a, b) =>
        (a as ClassPrediction).class.localeCompare((b as ClassPrediction).class)
      );
    else
      newPredictions.sort(
        (a, b) => (a as InstancePrediction).id - (b as InstancePrediction).id
      );
    return new PredictionDataset(
      newPredictions,
      this.modelNames,
      this.isClassLevel
    );
  }

  sortByComparisonMetric(
    comparisonName: string,
    modelID: string,
    rawValue: boolean,
    sortFn: (a: number, b: number) => number
  ): PredictionDataset {
    let newPredictions = Array.from(this.predictions);
    const scoreKey = rawValue ? "value" : "difference";
    if (
      newPredictions.length > 0 &&
      !newPredictions[0].comparisons[comparisonName][modelID].hasOwnProperty(
        scoreKey
      )
    ) {
      console.error(`Can't get key '${scoreKey}', missing from predictions`);
      return this;
    }
    newPredictions.sort((pred1, pred2) => {
      return sortFn(
        pred1.comparisons[comparisonName][modelID][scoreKey]!,
        pred2.comparisons[comparisonName][modelID][scoreKey]!
      );
    });
    return new PredictionDataset(
      newPredictions,
      this.modelNames,
      this.isClassLevel
    );
  }

  getPredictionSlice(startIndex: number, endIndex: number) {
    return new PredictionDataset(
      this.predictions.slice(startIndex, endIndex),
      this.modelNames,
      this.isClassLevel
    );
  }

  getComparisonMetrics(
    comparisonName: string,
    modelID: string,
    rawValue: boolean
  ): number[] {
    const scoreKey = rawValue ? "value" : "difference";
    if (
      this.predictions.length > 0 &&
      !this.predictions[0].comparisons[comparisonName][modelID].hasOwnProperty(
        scoreKey
      )
    ) {
      console.error(`Can't get key '${scoreKey}', missing from predictions`);
      return [];
    }
    return this.predictions.map(
      (p) => p.comparisons[comparisonName][modelID][scoreKey]!
    );
  }

  aggregateClassLabels(
    options: { [key: string]: { relative?: boolean } } = {}
  ): PredictionDataset {
    if (this.isClassLevel) return this;
    let classIdxMapping: Map<string, number> = new Map();
    let classResults: ClassPrediction[] = [];
    this.predictions.forEach((p) => {
      let pred = p as InstancePrediction;
      if (pred.classes !== undefined) {
        pred.classes.forEach((className) => {
          if (!classIdxMapping.has(className)) {
            classIdxMapping.set(className, classResults.length);
            // create an empty class prediction object
            classResults.push({
              class: className,
              predictions: Object.fromEntries(
                this.modelNames.map((m) => [m, []])
              ),
              comparisons: emptyComparisonObjectLike(pred.comparisons),
            });
          }

          // add the results of this prediction
          let classPred = classResults[classIdxMapping.get(className)!];
          this.modelNames.forEach((modelID) => {
            classPred.predictions[modelID].push(pred.predictions[modelID]);
            Object.keys(classPred.comparisons).forEach((compName) => {
              let v = pred.comparisons[compName][modelID];
              classPred.comparisons[compName][modelID].difference +=
                v.difference;
              if (v.value != undefined)
                classPred.comparisons[compName][modelID].value! += v.value;
            });
          });
        });
      }
    });

    // divide all results by the number of instances in each class
    classResults.forEach((result) => {
      Object.keys(result.comparisons).forEach((compName) => {
        this.modelNames.forEach((modelID) => {
          let n = result.predictions[modelID].length;
          let v = result.comparisons[compName][modelID];
          v.difference /= n;
          if (v.value != undefined) v.value /= n;
        });
        let baseModel = this.modelNames[0];
        if (
          !!options[compName] &&
          options[compName].relative &&
          result.comparisons[compName][baseModel].value != undefined
        ) {
          this.modelNames.forEach((modelID) => {
            result.comparisons[compName][modelID].difference /=
              result.comparisons[compName][baseModel].value!;
          });
        }
      });
    });

    return new PredictionDataset(classResults, this.modelNames, true);
  }

  overallAverageComparisonMetrics(): { n: number; comparisons: ComparisonSet } {
    if (this.predictions.length == 0) return { n: 0, comparisons: {} };
    let overallResults = emptyComparisonObjectLike(
      this.predictions[0].comparisons
    );
    let instanceCount = 0;
    this.predictions.forEach((pred) => {
      // multiply by the number of instances in this class if we are in
      // a class-level dataset
      let predInstances = this.isClassLevel
        ? (pred as ClassPrediction).predictions[this.modelNames[0]].length
        : 1;
      Object.entries(pred.comparisons).forEach(([compName, comparison]) => {
        this.modelNames.forEach((modelID) => {
          overallResults[compName][modelID].difference +=
            comparison[modelID].difference * predInstances;
          if (comparison[modelID].value != undefined)
            overallResults[compName][modelID].value! +=
              comparison[modelID].value! * predInstances;
        });
      });
      instanceCount += predInstances;
    });

    // divide all results by the number of instances in each class
    Object.keys(overallResults).forEach((compName) => {
      this.modelNames.forEach((modelID) => {
        let v = overallResults[compName][modelID];
        v.difference /= instanceCount;
        if (v.value != undefined) v.value /= instanceCount;
      });
    });

    return { n: instanceCount, comparisons: overallResults };
  }
}

// computes value counts for predictions in the dataset, optionally counting
// pairs of base -> model predictions
export function countPredictions(
  predictions: Prediction[],
  comparedTo: Prediction[] | null = null
): { pred: any; base?: any; count: number }[] {
  let hashFn = (pred: any, base: any): string => {
    if (!!comparedTo) return `${pred} ##--->## ${base}`;
    return `${pred}`;
  };

  let predictionCounts: { pred: any; base?: any; count: number }[] = [];
  let predictionIndexMap = new Map<string, number>();
  predictions.forEach((pred, i) => {
    let basePred = !!comparedTo ? comparedTo[i].pred : undefined;
    let hashableVersion = hashFn(pred.pred, basePred);
    if (!predictionIndexMap.has(hashableVersion)) {
      predictionIndexMap.set(hashableVersion, predictionCounts.length);
      predictionCounts.push({ pred: pred.pred, base: basePred, count: 0 });
    }
    predictionCounts[predictionIndexMap.get(hashableVersion)!].count += 1;
  });

  return predictionCounts.sort((a, b) => b.count - a.count);
}
