package MyClassDemo;

import java.util.Stack;

public class MinStack {

    Stack<Integer> stk1 = new Stack<>();
    Stack<Integer> stk2 = new Stack<>();
    int min = 2147483647;


    /** initialize your data structure here. */
    public MinStack() {
        stk2.push(min);
    }

    public void push(int x) {
        stk1.push(x);
        if (x < min){
            min = x;
        }
        stk2.push(min);
    }

    public void pop() {
        stk1.pop();
        stk2.pop();
        min = stk2.peek();
    }

    public int top() {
        return stk1.peek();
    }

    public int min() {
        return stk2.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.min();
 */
