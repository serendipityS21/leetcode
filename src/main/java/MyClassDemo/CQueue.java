package MyClassDemo;


import java.util.Stack;

/**
 * 剑指offer 09： 用两个栈实现队列
 * 用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )
 * <p>
 *  
 * <p>
 * 示例 1：
 * <p>
 * 输入：
 * ["CQueue","appendTail","deleteHead","deleteHead"]
 * [[],[3],[],[]]
 * 输出：[null,null,3,-1]
 * 示例 2：
 * <p>
 * 输入：
 * ["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
 * [[],[],[5],[2],[],[]]
 * 输出：[null,-1,null,null,5,2]
 * 提示：
 * <p>
 * 1 <= values <= 10000
 * 最多会对 appendTail、deleteHead 进行 10000 次调用
 */
public class CQueue {

    int val;
    Stack<Integer> stk1, stk2;

    public CQueue() {
        stk1 = new Stack<>();
        stk2 = new Stack<>();
        val = 0;

    }

    public void appendTail(int value) {

        //插入一个元素
        //stk1保存  底部存新插入的   顶部存老的
        while (!stk1.isEmpty()) {
            stk2.push(stk1.pop());
        }
        stk1.push(value);

        while (!stk2.isEmpty()) {
            stk1.push(stk2.pop());
        }

        val++;
    }

    public int deleteHead() {
        //删除队列首部元素
        //删除栈顶
        if (val == 0) {
            return -1;
        }

        int res = stk1.pop();
        val--;
        return res;
    }

    public void append(int val) {
        stk1.push(val);
        val++;
    }

    public int delete() {
        while (!stk1.isEmpty()) {
            stk2.push(stk1.pop());
        }
        int res = stk2.pop();
        val--;
        return res;
    }


}


/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue obj = new CQueue();
 * obj.appendTail(value);
 * int param_2 = obj.deleteHead();
 */
