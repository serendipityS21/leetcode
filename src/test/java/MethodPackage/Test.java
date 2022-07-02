package MethodPackage;

import MyClassDemo.InitDemo;
import MyClassDemo.TreeNode;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;


class Test {
    public static void main(String[] args) {
        Solution s = new Solution();
        TreeNode root = s.sortedArrayToBST2(new int[]{-10, -3, 0, 5, 9});
        String out = s.treeNodeToString(root);

        System.out.print(out);
    }


}














