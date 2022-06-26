package MethodPackage;

import MyClassDemo.ListNode;
import MyClassDemo.TreeNode;

import java.util.*;

public class OfferTwo {
    /**
     * 剑指 Offer 03. 数组中重复的数字
     */
    public int findRepeatNumber(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num)){
                return num;
            }else {
                set.add(num);
            }
        }
        return 0;
    }

    public int findRepeatNumber2(int[] nums) {
        int[] arr = new int[nums.length];
        for (int num : nums) {
            if (arr[num] == 1){
                return num;
            }else {
                arr[num] = 1;
            }
        }
        return 0;
    }

    /**
     * 剑指 Offer 04. 二维数组中的查找
     */
    //递归超时
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        return find2DHelper(matrix, 0, 0, target);
    }

    private boolean find2DHelper(int[][] matrix, int row, int col, int target) {
        if (row > matrix.length || col > matrix[0].length){
            return false;
        }else if (matrix[row][col] == target){
            return true;
        }else if (matrix[row][col] > target){
            return false;
        }
        return find2DHelper(matrix, row + 1, col, target) || find2DHelper(matrix, row, col + 1, target);
    }

    //非递归
    public boolean findNumberIn2DArray2(int[][] matrix, int target) {
        int i = matrix.length - 1;
        int j = 0;
        while (i >= 0 && j < matrix[0].length){
            if (matrix[i][j] == target){
                return true;
            }else if (matrix[i][j] > target){
                i--;
            }else {
                j++;
            }
        }
        return false;
    }

    /**
     * 剑指 Offer 05. 替换空格
     */
    public String replaceSpace(String s) {
        char[] chars = s.toCharArray();
        String temp = "%20";
        StringBuilder sb = new StringBuilder();
        for (char c : chars) {
            if (c == ' '){
                sb.append(temp);
            }else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * 剑指 Offer 06. 从尾到头打印链表
     */
    //先注入值
    public int[] reversePrint(ListNode head) {
        Stack<Integer> stack = new Stack<>();
        while (head != null){
            stack.push(head.val);
            head = head.next;
        }
        int[] ans = new int[stack.size()];
        int i = 0;
        while (!stack.isEmpty()){
            ans[i++] = stack.pop();
        }
        return ans;
    }

    //先注入结点
    public int[] reversePrint_2(ListNode head) {
        Stack<ListNode> stack = new Stack<>();
        ListNode p = head;
        while (p != null){
            stack.push(p);
            p = p.next;
        }
        int[] ans = new int[stack.size()];
        int i = 0;
        while (!stack.isEmpty()){
            ans[i++] = stack.pop().val;
        }
        return ans;
    }

    /**
     * 剑指 Offer 07. 重建二叉树
     */
    //利用原理,先序遍历的第一个节点就是根。在中序遍历中通过根 区分哪些是左子树的，哪些是右子树的
    //左右子树，递归
    int[] preorder;                                 //标记中序遍历
    Map<Integer, Integer> dic = new HashMap<>();    //保留的先序遍历
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        this.preorder = preorder;
        for (int i = 0; i < inorder.length; i++) {
            dic.put(inorder[i], i);
        }
        return recur(0, 0, inorder.length - 1);
    }

    /**
     * @param root   先序遍历的索引
     * @param left   中序遍历的索引
     * @param right  中序遍历的索引
     */
    private TreeNode recur(int root, int left, int right) {
        //相等就是自己
        if (left > right){
            return null;
        }
        //node在先序里面的
        TreeNode node = new TreeNode(preorder[root]);
        //有了先序的，再根据先序的，在中序中获取当前根的索引
        int i = dic.get(preorder[root]);
        //左子树的根节点就是 左子树的（前序遍历）第一个， 就是+1， 左边界就是left， 右边界就是当前根节点索引-1
        node.left = recur(root + 1, left, i - 1);
        //由根节点在中序遍历的idx 区分成2段,idx 就是根

        //右子树的根的索引为先序中的 当前根位置 + 左子树的数量 + 1
        //递归右子树的左边界为中序中当前根节点+1
        //递归右子树的右边界为中序中原来右子树的边界
        node.right = recur(root + i - left + 1, i + 1, right);
        return node;
    }

    /**
     * 剑指 Offer 09. 用两个栈实现队列
     */
    private Stack<Integer> A;
    private Stack<Integer> B;
    public void CQueue() {
        A = new Stack<>();
        B = new Stack<>();
    }

    public void appendTail(int value) {
        A.push(value);
    }

    public int deleteHead() {
        if(!B.isEmpty()){
            return B.pop();
        }else {
            while (!A.isEmpty()){
                B.push(A.pop());
            }
            return B.isEmpty() ? -1 : B.pop();
        }
    }

    /**
     * 剑指 Offer 10- I. 斐波那契数列
     */
    int[] cacheFib;
    public int fib(int n) {
        cacheFib = new int[n + 1];
        return fibHelper(n, cacheFib);
    }

    private int fibHelper(int n, int[] cache) {
        if (n <= 1){
            return n;
        }
        if (cache[n] == 0){
            cache[n] = (fibHelper(n - 1, cache) + fibHelper(n - 2, cache)) % 1000000007;
        }
        return cache[n];
    }

    /**
     * 剑指 Offer 10- II. 青蛙跳台阶问题
     */
    int[] cacheSteps;
    public int numWays(int n) {
        cacheSteps = new int[n + 1];
        return numWaysHelper(n, cacheSteps);
    }

    private int numWaysHelper(int n, int[] cache) {
        if (n <= 1){
            return 1;
        }
        if (cache[n] == 0){
            cache[n] = (numWaysHelper(n - 1, cache) + numWaysHelper(n - 2, cache)) % 1000000007;
        }
        return cache[n];
    }

    /**
     * 剑指 Offer 11. 旋转数组的最小数字
     */
    public int minArray(int[] numbers) {
        int i = 0;
        int j = numbers.length - 1;
        while (i < j){
            int mid = i + (j - i) / 2;
            if (numbers[mid] > numbers[j]){
                i = mid + 1;
            }else if (numbers[mid] < numbers[j]){
                j = mid;
            }else {
                j--;
            }
        }
        return numbers[i];
    }

    /**
     * 剑指 Offer 12. 矩阵中的路径
     */
    int[][] flagBoard;
    public boolean exist(char[][] board, String word) {
        flagBoard = new int[board.length][board[0].length];
        char[] chars = word.toCharArray();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (existHelper(board, chars, 0,i, j)){
                    return true;
                }
            }
        }
        return false;
    }

    private boolean existHelper(char[][] board, char[] chars, int index, int row, int col) {
        if (index == chars.length){
            return true;
        }
        if (row >= 0 && row < board.length && col >= 0 && col < board[0].length && flagBoard[row][col] != 1 && board[row][col] == chars[index]){
            flagBoard[row][col] = 1;
            boolean res =  existHelper(board, chars, index + 1, row + 1, col)
                    ||existHelper(board, chars, index + 1, row, col + 1)
                    ||existHelper(board, chars, index + 1, row, col - 1)
                    ||existHelper(board, chars, index + 1, row - 1, col);
            flagBoard[row][col] = 0;
            return res;
        }else {
            return false;
        }
    }

    /**
     * 剑指 Offer 13. 机器人的运动范围
     */

    //方法一： 深度优先遍历 DFS
    boolean[][] visited;
    int m, n, k;
    public int movingCount(int m, int n, int k) {
        this.m = m;
        this.n = n;
        this.k = k;
        this.visited = new boolean[m][n];
        return dfs(0, 0, 0, 0);
    }

    private int dfs(int row, int col, int rs, int cs) {
        if (row >= m || col >= n || k < rs + cs || visited[row][col]){
            return 0;
        }
        visited[row][col] = true;
        return 1 + dfs(row + 1, col, (row + 1) % 10 != 0 ? rs + 1 : rs - 8, cs)
                + dfs(row, col + 1, rs, (col + 1) % 10 != 0 ? cs + 1 : cs - 8);
    }

    //方法二： 广度优先算法 BFS
    public int movingCount_BFS(int m, int n, int k) {
        boolean[][] visited = new boolean[m][n];
        int res = 0;
        Queue<int[]> queue = new LinkedList<>();
        queue.add(new int[]{0, 0, 0, 0});
        while (!queue.isEmpty()){
            int[] x = queue.poll();
            int row = x[0], col = x[1], rs = x[2], cs = x[3];
            if (row >= m || col >= n || k < rs + cs || visited[row][col]){
                continue;
            }
            visited[row][col] = true;
            res++;
            queue.add(new int[]{row + 1, col, (row + 1) % 10 != 0 ? rs + 1 : rs - 8, cs});
            queue.add(new int[]{row, col + 1, rs, (col + 1) % 10 != 0 ? cs + 1 : cs - 8});
        }
        return res;
    }

    /**
     * 剑指 Offer 14- I. 剪绳子
     */
    public int cuttingRope(int n) {
        if (n <= 2){
            return 1;
        }
        if (n == 3){
            return 2;
        }
        if (n % 3 == 1){
            return (int) Math.pow(3, n / 3 - 1) * 4;
        }
        return (int) Math.pow(3, n / 3) * (n % 3 == 0 ? 1 : 2);
    }

    /**
     * 剑指 Offer 14- II. 剪绳子 II
     */
    public int cuttingRopeII(int n) {
        if (n <= 3){
            return n - 1;
        }
        long res = 1;
        final int p = 1000000007;
        while (n > 4){
            res = res * 3 % p;
            n -= 3;
        }
        return (int) (res * n % p);
    }

    /**
     * 剑指 Offer 15. 二进制中1的个数
     */
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int res = 0;
        while (n != 0){
            res++;
            n &= n - 1;
        }
        return res;
    }

    /**
     * 剑指 Offer 16. 数值的整数次方
     */
    public double myPow(double x, int n) {
        if (x == 0){
            return 0;
        }
        long b = n;
        double res = 1.0;
        if (b < 0){
            b *= -1;
            x = 1 / x;
        }
        while (b > 0){
            if ((b & 1) == 1){
                res *= x;
            }
            x *= x;
            b >>= 1;
        }
        return res;
    }

    /**
     * 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
     */
    public int[] exchange(int[] nums) {
        int low = 0;
        int high = nums.length - 1;
        while (low < high){
            while (low < high && nums[high] % 2 == 0){
                high--;
            }
            while (low < high && nums[low] % 2 != 0){
                low++;
            }
            int temp = nums[low];
            nums[low] = nums[high];
            nums[high] = temp;
        }
        return nums;
    }
}
