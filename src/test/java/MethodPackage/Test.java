package MethodPackage;

import MyClassDemo.InitDemo;


class Test {

    public static void main(String[] args) {
        Solution s = new Solution();
        InitDemo it = new InitDemo();


        String[] tokens = {"5","1","-","7","+"};
        int res = s.evalRPN(tokens);
        System.out.println(res);


    }


}














