package MethodPackage;

import MyClassDemo.InitDemo;
import MyClassDemo.TreeNode;

import java.util.Arrays;
import java.util.List;


class Test {

    public static void main(String[] args) {
        Solution s = new Solution();
        InitDemo it = new InitDemo();


        TreeNode root = it.convert(new Integer[]{5,4,8,11,null,13,4,7,2,null,null,null,1});
        boolean res = s.hasPathSum2(root, 22);

        System.out.println(res);


    }


}














