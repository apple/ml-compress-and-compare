<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
  
<!--
  @component
  Generates a hover tooltip. It creates a slot with an exposed variable via `let:detail` that contains information about the event. Use the slot to populate the body of the tooltip using the exposed variable `detail`.
 -->
<script lang="ts">
  import type {
    Model,
    CompressionOperation,
    CompressionOperationSpec,
    OperationParameterSpec,
    Metric,
  } from "$lib/data_structures/model_map";
  import * as d3 from "d3";

  export let operations: CompressionOperationSpec[] = [];
  export let metrics: Metric[] = [];
  export let models: Model[] = [];

  export let model: Model;

  function toTitleCase(string: string) {
    return string
      .split(" ")
      .map((word) => word[0].toUpperCase() + word.substr(1))
      .join(" ");
  }

  function formatOperationParameter(
    operation: CompressionOperation,
    parameterName: string,
    parameterValue: any
  ): string {
    let operationSpec:
      | { parameters: { [key: string]: OperationParameterSpec } }
      | undefined = !!operation
      ? operations.find((op) => op.name == operation!.name)
      : { parameters: {} };

    if (!operationSpec) return parameterValue;

    let paramSpec = operationSpec.parameters[parameterName];
    if (!paramSpec || typeof parameterValue === "string") return parameterValue;
    let formatter = d3.format(paramSpec.format || ".2g");
    return formatter(parameterValue);
  }

  function formatMetric(metricName: string, metricValue: any): string {
    let metricSpec: Metric | undefined = metrics.find(
      (met) => met.name == metricName
    );

    if (!metricSpec || typeof metricValue === "string") return metricValue;

    let formatter = d3.format(metricSpec.format || ".2g");
    return formatter(metricValue) + (metricSpec.unit ? metricSpec.unit : "");
  }

  let formatChange = d3.format("+.1~%");

  let metricsToShow: { [key: string]: any } = {};
  $: if (!!model) {
    let hasPrimary = metrics.find((m) => m.primary);
    metricsToShow = Object.fromEntries(
      Object.entries(model.metrics).filter(([metricName, _]) => {
        if (!metrics) return true;
        let metricSpec: Metric | undefined = metrics.find(
          (met) => met.name == metricName
        );
        return !hasPrimary || (!!metricSpec && metricSpec.primary);
      })
    );
  }

  let parentModel: Model | null = null;
  $: if (models.length > 0 && model.base) {
    parentModel = models.find((m) => m.id == model.base) || null;
  } else {
    parentModel = null;
  }
</script>

{#if model != null}
  <div class="w-full text-sm">
    <div>
      <!--- TOOD: Automate what is shown in the tooltip. Currently too many to show.-->
      <div class="row">
        <p class="text-gray-400 font-bold text-right">OPERATION</p>
        <p class="text-gray-700 font-mono">
          {model.operation != null ? model.operation.name : "None"}
        </p>
      </div>
      {#if model.operation != null}
        {#each Object.entries(model.operation.parameters) as [paramName, paramValue]}
          <div class="row">
            <p class="text-gray-400 font-bold text-right">
              {paramName.toUpperCase()}
            </p>
            <p class="text-gray-700 font-mono">
              {formatOperationParameter(model.operation, paramName, paramValue)}
            </p>
          </div>
        {/each}
      {/if}
      <div class="mt-2" />
      <div class="text-gray-500 row text-xs p-1 mb-1 rounded-full bg-gray-100">
        <p class="font-bold text-right">Metric</p>
        <p class="font-bold">
          Value
          {#if !!parentModel}
            (vs. parent)
          {/if}
        </p>
      </div>
      {#each Object.entries(metricsToShow) as [metricName, metricValue]}
        <div class="row">
          <p class="text-gray-400 font-bold text-right">
            {metricName.toUpperCase()}
          </p>
          <p class="text-gray-700 font-mono">
            {formatMetric(metricName, metricValue)}
            {#if !!parentModel && parentModel.metrics.hasOwnProperty(metricName) && Math.abs((metricValue - parentModel.metrics[metricName]) / parentModel.metrics[metricName]) >= 0.01}
              <span class="text-gray-500">
                {#if parentModel.metrics[metricName] != 0}
                  ({formatChange(
                    (metricValue - parentModel.metrics[metricName]) /
                      parentModel.metrics[metricName]
                  )})
                {:else}
                  (&infin;)
                {/if}
              </span>
            {/if}
          </p>
        </div>
      {/each}
    </div>
  </div>
{/if}

<style>
  .row {
    display: grid;
    grid-template-columns: 50% auto;
    gap: 16px;
  }
</style>
