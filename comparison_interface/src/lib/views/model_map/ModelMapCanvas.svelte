<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
  
<script lang="ts">
  import { getContext, onMount } from "svelte";
  import { scaleCanvas } from "layercake";
  import type { Readable } from "svelte/store";
  import { brightnessByColor, type Position } from "$lib/utils";
  import * as d3 from "d3";
  import type ModelMapState from "./ModelMapState.svelte";
  import type { MarkSet } from "$lib/chart_components/animated_marks";

  export let zoomTransform: d3.ZoomTransform = d3.zoomIdentity;

  export let state: ModelMapState;

  export let needsDraw = false;
  let _needsDrawOnce = false;

  export let marks: MarkSet | null = null;

  export let hoveredIDs: string[] = [];
  export let showCrosshair = false;

  const { width, height } = getContext("LayerCake") as {
    width: Readable<number>;
    height: Readable<number>;
  };
  const { ctx } = getContext("canvas") as {
    ctx: Readable<CanvasRenderingContext2D>;
  };

  function nodeIsInCanvas(
    position: [number, number],
    nodeSize: [number, number],
    w: number,
    h: number
  ): boolean {
    return (
      position[0] + nodeSize[0] * 0.5 >= 0 &&
      position[0] - nodeSize[0] * 0.5 <= w &&
      position[1] + nodeSize[1] * 0.5 >= 0 &&
      position[1] - nodeSize[1] * 0.5 <= h
    );
  }

  const bezierFraction = 0.6;
  const bezierLeftFlatFraction = 0.0;
  const dotPadding = 0.1;

  function drawHoverCrosshair() {
    if (hoveredIDs.length != 1) return;
    let hoveredID = hoveredIDs[0];

    $ctx.save();
    $ctx.strokeStyle = "#9ca3af";
    $ctx.setLineDash([4, 4]);
    let hoveredMark = marks?.getMarkByID(hoveredID);
    let coord: [number, number] = [
      hoveredMark.attr("x"),
      hoveredMark.attr("y"),
    ];
    $ctx.moveTo(coord[0], 0);
    $ctx.lineTo(coord[0], $height);
    $ctx.moveTo(0, coord[1]);
    $ctx.lineTo($width, coord[1]);
    $ctx.stroke();
    $ctx.restore();
  }

  function drawEdges() {
    $ctx.strokeStyle = "#f2f2f2"; // "#d1d5db";
    $ctx.lineWidth = 1;

    marks?.decorations.forEach((edge: any) => {
      $ctx.save();
      $ctx.globalAlpha =
        Math.min(edge.marks[0].attr("alpha"), edge.marks[1].attr("alpha")) *
        0.8;

      let startRadius = edge.marks[0].attr("radius");
      let endRadius = edge.marks[1].attr("radius");
      let radiusFactor = 0.5;

      let startX = edge.attr("x") + startRadius - 2;
      let startY = edge.attr("y");
      let endX = edge.attr("x2") - endRadius + 2;
      let endY = edge.attr("y2");

      let firstLineWidth =
        (startRadius * (1 - bezierFraction) + endRadius * bezierFraction) *
        radiusFactor;
      let secondLineWidth =
        (startRadius * bezierFraction + endRadius * (1 - bezierFraction)) *
        radiusFactor;

      const grd = $ctx.createLinearGradient(startX, startY, endX, endY);
      grd.addColorStop(0, edge.marks[0].attr("fillStyle"));
      grd.addColorStop(1, edge.marks[1].attr("fillStyle"));

      // Fill with gradient
      $ctx.fillStyle = grd;

      $ctx.beginPath();
      $ctx.moveTo(startX, startY + startRadius * radiusFactor);
      $ctx.bezierCurveTo(
        startX + (endX - startX) * bezierFraction - firstLineWidth,
        startY + firstLineWidth,
        endX -
          (endX - startX) * (bezierLeftFlatFraction + bezierFraction) -
          secondLineWidth,
        endY + secondLineWidth,
        endX,
        endY + endRadius * radiusFactor
      );
      $ctx.lineTo(endX, endY - endRadius * radiusFactor);
      $ctx.bezierCurveTo(
        endX -
          (endX - startX) * (bezierLeftFlatFraction + bezierFraction) +
          secondLineWidth,
        endY - secondLineWidth,
        startX + (endX - startX) * bezierFraction + firstLineWidth,
        startY - firstLineWidth,
        startX,
        startY - startRadius * radiusFactor
      );
      $ctx.lineTo(startX, startY + startRadius * radiusFactor);
      $ctx.fill();
      $ctx.stroke();
      $ctx.restore();
    });
  }

  function drawNodes() {
    $ctx.lineWidth = 4 / zoomTransform.k;
    marks!.forEach((mark) => {
      let alpha = mark.attr("alpha");
      if (alpha <= 0.001) return;

      let x = mark.attr("x");
      let y = mark.attr("y");
      let width = mark.attr("width");
      let height = mark.attr("height");
      let radius = mark.attr("radius");

      if (!nodeIsInCanvas([x, y], [width, height], $width, $height)) return;

      let fillStyle = mark.attr("fillStyle");
      let strokeStyle = mark.attr("strokeStyle");
      let lineWidth = mark.attr("lineWidth");

      $ctx.save();

      $ctx.globalAlpha = alpha;

      /*$ctx.fillStyle = "#e5e7eb";
      $ctx.beginPath();
      $ctx.roundRect(
        x - width * 0.5,
        y - height * 0.5,
        width,
        height,
        Math.min(width, height) * 0.2
      );
      $ctx.closePath();
      $ctx.fill();*/

      /*let icon = mark.attr("icon");
      if (!!icon) {
        $ctx.fillStyle = "#9ca3af";
        $ctx.font = "12pt sans-serif";
        $ctx.fillText(icon, x - width * 0.5 + dotPadding * width, y + 6);
        console.log(icon);
      }*/

      $ctx.fillStyle = fillStyle;
      $ctx.beginPath();
      $ctx.ellipse(x, y, radius, radius, 0, 0, 2 * Math.PI, true);
      $ctx.closePath();
      $ctx.fill();
      if (brightnessByColor(fillStyle) < 128) $ctx.strokeStyle = "#f2f2f2";
      else $ctx.strokeStyle = "#d1d5db";
      $ctx.lineWidth = 1;
      $ctx.stroke();

      if (lineWidth > 0) {
        $ctx.beginPath();
        $ctx.ellipse(
          x,
          y,
          radius + lineWidth * 0.5,
          radius + lineWidth * 0.5,
          0,
          0,
          2 * Math.PI,
          true
        );
        $ctx.closePath();
        $ctx.strokeStyle = strokeStyle;
        $ctx.lineWidth = lineWidth;
        $ctx.stroke();
      }

      $ctx.restore();

      let text = mark.attr("parameterText") as { [key: string]: string } | null;
      if (!!text && Object.entries(text).length > 0) {
        let alpha = mark.attr("labelAlpha");
        if (alpha > 0.0) {
          $ctx.save();
          $ctx.fillStyle = "#9ca3af";
          $ctx.strokeStyle = "rgb(255, 255, 255, 0.8)";
          $ctx.lineWidth = 6.0;
          $ctx.globalAlpha = alpha;
          $ctx.textAlign = "right";
          $ctx.textBaseline = "bottom";
          $ctx.font =
            'bold 10pt ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"';

          let parameterNames = Object.keys(text).sort();
          let metrics = $ctx.measureText(parameterNames[0]);
          let bottomY = y - height * 0.5 - 12;
          let lineHeight = 12;
          parameterNames
            .slice()
            .reverse()
            .forEach((param) => {
              $ctx.strokeText(param.toLocaleUpperCase(), x - 4, bottomY);
              $ctx.fillText(param.toLocaleUpperCase(), x - 4, bottomY);
              bottomY -= lineHeight + 4;
            });

          $ctx.fillStyle = "#374151";
          $ctx.textAlign = "left";
          $ctx.font =
            'normal 10pt ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace';

          metrics = $ctx.measureText(text[parameterNames[0]]);
          bottomY = y - height * 0.5 - 12;
          parameterNames
            .slice()
            .reverse()
            .forEach((param) => {
              $ctx.strokeText(text[param], x + 4, bottomY);
              $ctx.fillText(text[param], x + 4, bottomY);
              bottomY -= lineHeight + 4;
            });
          $ctx.restore();
        }
      }
    });
  }

  function draw() {
    if (!$ctx || !marks) return;
    console.log("drawing");
    scaleCanvas($ctx, $width, $height);
    $ctx.clearRect(0, 0, $width, $height);
    $ctx.translate(0.5, 0.5);

    // Hovered row/column
    if (hoveredIDs.length == 1 && showCrosshair) {
      drawHoverCrosshair();
    }

    $ctx.save();

    // Edges
    drawEdges();

    // Nodes
    drawNodes();

    $ctx.restore();
  }

  let timer: d3.Timer;

  function setupTimer() {
    timer = d3.timer(() => {
      if (needsDraw || _needsDrawOnce) draw();
      _needsDrawOnce = false;
    });
  }

  onMount(() => {
    setupTimer();
    draw();
  });
</script>
