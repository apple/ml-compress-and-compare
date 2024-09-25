<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import { scaleCanvas } from "layercake";
  import { getContext } from "svelte";
  import type { Readable } from "svelte/store";

  export let multiselectRegion: Rectangle | null = null;

  const { width, height } = getContext("LayerCake") as {
    width: Readable<number>;
    height: Readable<number>;
  };
  const { ctx } = getContext("canvas") as {
    ctx: Readable<CanvasRenderingContext2D>;
  };

  function drawMultiselectRegion() {
    if (!multiselectRegion) return;

    $ctx.save();
    $ctx.fillStyle = "#93c5fd44";
    $ctx.strokeStyle = "#93c5fd";
    $ctx.lineWidth = 2;
    $ctx.setLineDash([4, 4]);

    $ctx.beginPath();
    $ctx.roundRect(
      multiselectRegion.x,
      multiselectRegion.y,
      multiselectRegion.width,
      multiselectRegion.height,
      4
    );
    $ctx.closePath();
    $ctx.fill();
    $ctx.stroke();
  }

  function draw() {
    if (!$ctx) return;
    scaleCanvas($ctx, $width, $height);
    $ctx.clearRect(0, 0, $width, $height);
    $ctx.translate(0.5, 0.5);

    if (!!multiselectRegion) drawMultiselectRegion();
  }

  $: $ctx, multiselectRegion, draw();
</script>
