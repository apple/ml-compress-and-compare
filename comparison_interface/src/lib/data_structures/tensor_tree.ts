/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2024 Apple Inc. All Rights Reserved.
 */

export interface TensorTreeNode<TensorDataType = any> {
  parent?: string;
  children: string[];
  type: string;
  data: { [key: string]: TensorDataType };
}

export interface TensorTree<TensorDataType = any> {
  nodes: { [key: string]: TensorTreeNode<TensorDataType> };
  type: string;
}
