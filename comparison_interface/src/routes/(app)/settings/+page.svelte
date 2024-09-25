<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->
   
 <script lang="ts">
  import AxisX from "$lib/chart_components/AxisX.svelte";
  import {
    SocketDataSource,
    getModelServerURL,
    type DataSource,
    setModelServerURL,
  } from "$lib/datasource";
  import { onMount } from "svelte";
  import { base as baseURL } from "$app/paths";

  let modelServerURL: string = "";

  onMount(() => {
    modelServerURL = getModelServerURL();
  });

  let validURL: boolean = true;
  let errorMessage: string = "";

  $: {
    validURL = /^(ftp|http|https):\/\/[^ "]+$/.test(modelServerURL);
    if (validURL) errorMessage = "";
    else errorMessage = "Enter a valid URL.";
  }

  let testDataSource: DataSource | null = null;

  function attemptConnect() {
    console.log("attempting connect");
    testDataSource = new SocketDataSource(modelServerURL, "/model_map", 0);
    testDataSource.onAttach(() => {
      // Success
      console.log("successfully connected!");
      testDataSource = null;
      setModelServerURL(modelServerURL);
      window.location.href = baseURL + "/";
    });
    testDataSource.onDetach((model, error) => {
      if (error)
        errorMessage = `Model server could not be reached at this URL (${error})`;
      if (!!testDataSource) testDataSource!.disconnect();
      testDataSource = null;
    });
    setTimeout(() => {
      testDataSource!.connect();
    }, 1000);
    return false;
  }
</script>

<div class="w-full h-full bg-gray-100 flex items-center justify-center">
  <div class="w-1/2 p-4 bg-white rounded-lg" style="max-width: 600px;">
    <h2 class="text-lg mb-4">Connect to Model Server</h2>
    <p class="text-sm text-gray-600 mb-4">
      Enter a URL (such as <span class="font-mono">http://localhost:PORT</span>)
      that is accessible from your local machine and exposes a server created
      using the
      <span class="font-mono">interactive-compression</span> Python package.
    </p>
    <form on:submit|preventDefault={attemptConnect}>
      <input
        class="w-full font-mono"
        type="text"
        bind:value={modelServerURL}
        placeholder="http://localhost:5001"
      />
      {#if errorMessage.length > 0}
        <p class="mt-2 text-sm text-red-500">{errorMessage}</p>
      {/if}
      <div class="mt-4 flex justify-center w-full">
        {#if testDataSource == null}
          <input
            type="submit"
            disabled={!validURL}
            class="primary-btn cursor-pointer"
            value="Connect"
          />
        {:else}
          <div class="text-sm">Connecting...</div>
          <svg
            aria-hidden="true"
            class="ml-4 w-6 h-6 text-gray-400 animate-spin fill-blue-600"
            viewBox="0 0 100 101"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
              fill="currentColor"
            />
            <path
              d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
              fill="currentFill"
            />
          </svg>
        {/if}
      </div>
    </form>
  </div>
</div>
