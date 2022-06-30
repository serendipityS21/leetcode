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
        TreeNode root1 = s.stringToTreeNode("[1,3,2,5]");

        TreeNode root2 = s.stringToTreeNode("[2,1,3,null,4,null,7]");

        TreeNode ret = new Solution().mergeTrees2(root1, root2);

        String out = s.treeNodeToString(ret);

        System.out.print(out);
    }


}














