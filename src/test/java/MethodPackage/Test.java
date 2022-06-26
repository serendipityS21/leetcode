package MethodPackage;

import MyClassDemo.InitDemo;

import java.util.Arrays;


class Test {

    public static void main(String[] args) {
        Solution s = new Solution();
        InitDemo it = new InitDemo();


        int[] nums = {1,3,-1,-3,5,3,6,7};
        int[] res = s.maxSlidingWindow239(nums, 3);
        System.out.println(Arrays.toString(res));


    }


}














