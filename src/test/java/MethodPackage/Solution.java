package MethodPackage;


import MyClassDemo.ListNode;
import MyClassDemo.Node;
import MyClassDemo.TreeNode;

import java.util.*;

public class Solution {
    /**
     * 剑指offer 2: 对称的二叉树
     * 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
     * <p>
     * 例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
     */
    public boolean isSymmetric(TreeNode root) {
        return checkRoot(root, root);
    }
    private boolean checkRoot(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        return p.val == q.val && checkRoot(p.left, q.right) && checkRoot(p.right, q.left);
    }
    public boolean isSymmetric1(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> tmp = new ArrayList<>();

            int currentLevelCount = queue.size();
            for (int i = 0; i < currentLevelCount; i++) {
                TreeNode node = queue.poll();

                if (node != null) {
                    tmp.add(node.val);
                    queue.offer(node.left);
                    queue.offer(node.right);
                } else {
                    tmp.add(null);
                }
            }
            for (int i = 0; i < tmp.size() / 2; i++) {
                if (tmp.get(i) != tmp.get(tmp.size() - i - 1)) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 剑指offer 03:找出数组中重复的数字。
     * 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
     * <p>
     * 示例 1：
     * <p>
     * 输入：
     * [2, 3, 1, 0, 2, 5, 3]
     * 输出：2 或 3
     *  
     * <p>
     * 限制：
     * <p>
     * 2 <= n <= 100000
     */
    public int findRepeatNumber(int[] nums) {
        Set<Integer> set = new HashSet<>();
        int res = -1;
        for (int num : nums) {
            if (!set.add(num)) {
                res = num;
                break;
            }
        }
        return res;
    }

    /**
     * 剑指offer 04:二维数组中的查找
     * 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
     * <p>
     *  
     * <p>
     * 示例:
     * <p>
     * 现有矩阵 matrix 如下：
     * <p>
     * [
     * [1,   4,  7, 11, 15],
     * [2,   5,  8, 12, 19],
     * [3,   6,  9, 16, 22],
     * [10, 13, 14, 17, 24],
     * [18, 21, 23, 26, 30]
     */
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if ((matrix == null || matrix.length == 0) || matrix.length == 0 && matrix[0].length == 0) {
            return false;
        }
        int i = 0;
        int j = matrix[0].length - 1;    //ij对应右上角
        while (i < matrix.length && j >= 0) {
            //先判断是否相等，否则ij可能出现越界
            if (target == matrix[i][j]) {
                return true;
            } else if (target < matrix[i][j]) {
                j--;
            } else if (target > matrix[i][j]) {
                i++;
            }
        }
        return false;
    }

    /**
     * 剑指offer 05：替换空格
     * 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
     * <p>
     *  
     * <p>
     * 示例 1：
     * <p>
     * 输入：s = "We are happy."
     * 输出："We%20are%20happy."
     */
    public String replaceSpace(String s) {
        StringBuffer sb = new StringBuffer();
        sb.append(s);
        for (int i = 0; i < sb.length(); i++) {
            if ((sb.charAt(i)) == ' ') {
                sb.replace(i, i + 1, "%20");
            }
        }
        return sb.toString();
    }

    public String replaceSpace2(String s) {
        if (s == null || s.length() == 0){
            return s;
        }
        StringBuilder str = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c == ' '){
                str.append("  ");
            }
        }
        if (str.length() == 0){
            return s;
        }
        int left = s.length() - 1;
        s += str.toString();
        int right = s.length() - 1;
        char[] chars = s.toCharArray();
        while (left >= 0){
            if (chars[left] == ' '){
                chars[right--] = '0';
                chars[right--] = '2';
                chars[right] = '%';
            }else {
                chars[right] = chars[left];
            }
            left--;
            right--;
        }
        return new String(chars);
    }

    /**
     * 剑指offer 06:从尾到头打出链表
     */
    public int[] reversePrint(ListNode head) {
        Stack<ListNode> stack = new Stack<>();
        ListNode p = head;

        while (p.next != null) {
            p = p.next;
            stack.push(p);
        }

        int length = stack.size();
        int[] nums = new int[length];

        for (int i = 0; i < length; i++) {
            nums[i] = stack.pop().val;
        }
        return nums;
    }

    /**
     * 剑指 Offer 07. 重建二叉树
     * 输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。
     *
     * 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder.length == 0 || inorder.length == 0){
            return null;
        }else if(preorder.length == 1 && preorder.length == 1){
            TreeNode p = new TreeNode();
            p.val = preorder[0];
            p.left = null;
            p.right = null;
            return p;
        }
        int i = 0;
        TreeNode root = new TreeNode();
        TreeNode p;
        while (preorder[0] != inorder[i]){
            i++;
        }
        root.val = preorder[0];
        root.left = buildTree(Arrays.copyOfRange(preorder,1, i + 1), Arrays.copyOfRange(inorder, 0, i));
        root.right = buildTree(Arrays.copyOfRange(preorder,i + 1, preorder.length), Arrays.copyOfRange(inorder, i + 1, inorder.length));
        return root;
    }

    /**
     * 剑指offer 10-1 斐波那契数列
     *
     * @param n
     * @return
     */
    public int fib(int n) {
        final int mod = 1000000007;
        if (n < 2) {
            return n;
        } else {
            int p = 0;
            int q = 0;
            int r = 1;
            for (int i = 2; i <= n; i++) {
                p = q;
                q = r;
                r = (p + q) % mod;
            }
            return r;
        }
    }

    /**
     * 剑指offer 10-2 青蛙跳台问题
     *
     * @param n
     * @return
     */
    public int numWays(int n) {
        final int mod = 1000000007;
        if (n < 2) {
            return 1;
        } else {
            int p = 0;
            int q = 1;
            int r = 1;
            for (int i = 2; i <= n; i++) {
                p = q;
                q = r;
                r = (p + q) % mod;
            }
            return r;
        }
    }

    /**
     * 剑指offer 11 旋转数组的最小数字
     *
     * @param numbers
     * @return
     */
    public int minArray(int[] numbers) {
        int i = 0;
        int j = numbers.length - 1;
        while (i < j) {
            if (numbers[i] <= numbers[i + 1] && numbers[j] >= numbers[j - 1]) {
                i++;
                j--;
            } else if (numbers[i] <= numbers[i + 1]) {
                return numbers[j];
            } else {
                return numbers[i + 1];
            }
        }
        return numbers[0];
    }

    /**
     * 剑指offer 12 矩阵中的路径
     */
    //回溯法
    public boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (dfs(board, word, 0, i, j)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(char[][] board, String word, int u, int x, int y) {
        if (x >= board.length || x < 0 || y >= board[0].length || y < 0 || board[x][y] != word.charAt(u)) {
            return false;
        }
        if (u == word.length() - 1) {
            return true;
        }

        char temp = board[x][y];
        board[x][y] = '*';

        //递归遍历
        boolean res = dfs(board, word, u + 1, x - 1, y) || dfs(board, word, u + 1, x + 1, y)
                || dfs(board, word, u + 1, x, y - 1) || dfs(board, word, u + 1, x, y + 1);
        board[x][y] = temp;
        return res;
    }

    /**
     * 剑指 Offer 13. 机器人的运动范围
     * 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，
     * 它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。
     * 例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。
     * 请问该机器人能够到达多少个格子？
     */
    public int movingCount(int m, int n, int k) {
        if (k == 0){
            return 1;
        }
        Queue<int[]> queue = new LinkedList<>();
        int[] dx = {0, 1};
        int[] dy = {1, 0};
        boolean[][] vis = new boolean[m][n];
        vis[0][0] = true;
        queue.offer(new int[]{0,0});
        int ans = 1;
        while (!queue.isEmpty()){
            int[] cell = queue.poll();
            int x = cell[0];
            int y = cell[1];
            for (int i = 0; i < 2; i++) {
                int tx = dx[i] + x;
                int ty = dy[i] + y;
                if (tx < 0 || tx >= m || ty < 0 || ty >= n || vis[tx][ty] || getC(tx) + getC(ty) > k){
                    continue;
                }
                queue.offer(new int[]{tx, ty});
                vis[tx][ty] = true;
                ans++;
            }
        }
        return ans;
    }

    private int getC(int x){
        int res = 0;
        while (x != 0){
            res += x % 10;
            x /= 10;
        }
        return res;
    }

    /**
     * 剑指 Offer 14- I. 剪绳子
     * 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），
     * 每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？
     * 例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
     */
    public int cuttingRope_I(int n) {
        int sum = 1;
        if (n == 1 || n == 2){
            return 1;
        }else if (n == 3){
            return 2;
        }else {
            while (n > 4){
                sum *= 3;
                n -= 3;
            }
        }
        return sum * n;
    }

    /**
     * 剑指 Offer 14- II. 剪绳子 II
     * 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m段（m、n都是整数，n>1并且m>1），
     * 每段绳子的长度记为 k[0],k[1]...k[m - 1] 。请问 k[0]*k[1]*...*k[m - 1] 可能的最大乘积是多少？
     * 例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
     *
     * 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
     */
    public int cuttingRope_II(int n) {
        if (n <= 3) return n - 1;
        int b = n % 3;
        final int p = 1000000007;
        long rem = 1;
        long x = 3;
        for(int a = n / 3 - 1; a > 0; a /= 2) {
            if(a % 2 == 1) rem = (rem * x) % p;
            x = (x * x) % p;
        }
        if (b == 0) return (int) (rem * 3 % p);
        if (b == 1) return (int) (rem * 4 % p);
        return (int) (rem * 6 % p);
    }


    /**
     * 剑指 Offer 15. 二进制中1的个数
     * 编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量).）。
     *
     * 提示：
     *
     * 请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
     * 在 Java 中，编译器使用 二进制补码 记法来表示有符号整数。因此，在上面的示例 3中，输入表示有符号整数 -3。
     */
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0){
            n &= n - 1;
            count++;
        }
        return count;
    }

    /**
     * 剑指 Offer 16. 数值的整数次方
     * 实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题
     */
    public double myPow(double x, int n) {
        if (n == 0 || x == 1 || x == -1){
            if (x == -1 && n % 2 != 0){
                return -1;
            }
            return 1;
        }else if(x == 0 || (x > 1 || x < -1) && n == Integer.MIN_VALUE || (x < 1 && x > -1) && n == Integer.MAX_VALUE){
            return 0;
        }
        if (n < 0){
            n = Math.abs(n);
            x = 1 / x;
        }
        double pow = x;
        for (int i = 1; i < n; i++) {
            x *= pow;
        }
        return x;
    }
    public double myPow2(double x, int n){
        long N = n;
        return N >= 0 ? quickMul(x, N) : 1.0 / quickMul(x, -N);
    }
    private double quickMul(double x, long N){
        if (N == 0){
            return 1.0;
        }
        double y = quickMul(x, N / 2);
        return N % 2 == 0 ? y * y : y * y * x;
    }

    /**
     * 剑指 Offer 17. 打印从1到最大的n位数
     * 输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999
     */
    public int[] printNumbers(int n) {
        int t = 1;
        for (int i = 0; i < n; i++) {
            t *= 10;
        }
        int[] ans = new int[t - 1];
        for (int i = 0; i < t - 1; i++) {
            ans[i] = i + 1;
        }
        return ans;
    }

    /**
     * 剑指 Offer 18. 删除链表的节点
     * 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
     *
     * 返回删除后的链表的头节点
     */
    public ListNode deleteNode(ListNode head, int val) {
        if (head == null){
            return null;
        }
        ListNode p  = new ListNode();
        p.next = head;
        head = p;
        while (p.next != null){
            if (p.next.val == val){
                p.next = p.next.next;
            }else {
                p = p.next;
            }
        }
        return head.next;
    }

    /**
     * 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
     *
     * 数值（按顺序）可以分成以下几个部分：
     *
     * 若干空格
     * 一个小数或者整数
     * （可选）一个'e'或'E'，后面跟着一个整数
     * 若干空格
     * 小数（按顺序）可以分成以下几个部分：
     *
     * （可选）一个符号字符（'+' 或 '-'）
     * 下述格式之一：
     * 至少一位数字，后面跟着一个点 '.'
     * 至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
     * 一个点 '.' ，后面跟着至少一位数字
     * 整数（按顺序）可以分成以下几个部分：
     *
     * （可选）一个符号字符（'+' 或 '-'）
     * 至少一位数字
     * 部分数值列举如下：
     *
     * ["+100", "5e2", "-123", "3.1416", "-1E-16", "0123"]
     * 部分非数值列举如下：
     *
     * ["12e", "1a3.14", "1.2.3", "+-5", "12e+5.4"]
     */
    public boolean isNumber(String s) {
        List<char[]> list = new LinkedList<>();
        for (String c : s.split(", ")) {
            list.add(c.substring(1,c.length() - 1).trim().toCharArray());
        }
        for (char[] ch : list) {
            //小数点旗帜
            boolean dotFlag = true;
            //e / E旗帜
            boolean eEFlag = false;
            //number 旗帜
            boolean nFlag = true;
            //num旗帜 标志是否存在数字
            boolean numFlag = false;
            for (int i = 0; i < ch.length; i++) {
                if (ch[i] >= '0' && ch[i] <= '9'){
                    numFlag = true;
                    if (nFlag){
                        eEFlag = true;
                        nFlag = false;
                    }
                }else if (ch[i] == '+' || ch[i] == '-'){
                    if (i == 0){
                        numFlag = false;
                        continue;
                    }else if (ch[i - 1] == 'e' || ch[i - 1] == 'E'){
                        continue;
                    }else {
                        return false;
                    }
                }else if (ch[i] == 'E' || ch[i] == 'e'){
                    if (eEFlag){
                        dotFlag = false;
                        eEFlag = false;
                    }else {
                        return false;
                    }
                    numFlag = false;
                }else if (ch[i] == '.'){
                    if (dotFlag){
                        dotFlag = false;
                    }else {
                        return false;
                    }
                    numFlag = false;
                }else {
                    return false;
                }
            }
            if (!numFlag){
                return false;
            }
        }
        return true;
    }

    /**
     * 剑指 Offer 19. 正则表达式匹配
     * 请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。
     * 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。
     *
     * 示例 1:
     */
    public boolean isMatch(String A, String B) {
        int n = A.length();
        int m = B.length();
        boolean[][] f = new boolean[n + 1][m + 1];

        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                //分成空正则和非空正则两种
                if (j == 0) {
                    f[i][j] = i == 0;
                } else {
                    //非空正则分为两种情况 * 和 非*
                    if (B.charAt(j - 1) != '*') {
                        if (i > 0 && (A.charAt(i - 1) == B.charAt(j - 1) || B.charAt(j - 1) == '.')) {
                            f[i][j] = f[i - 1][j - 1];
                        }
                    } else {
                        //碰到 * 了，分为看和不看两种情况
                        //不看
                        if (j >= 2) {
                            f[i][j] |= f[i][j - 2];
                        }
                        //看
                        if (i >= 1 && j >= 2 && (A.charAt(i - 1) == B.charAt(j - 2) || B.charAt(j - 2) == '.')) {
                            f[i][j] |= f[i - 1][j];
                        }
                    }
                }
            }
        }
        return f[n][m];
    }

    /**
     * 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
     * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数在数组的前半部分，所有偶数在数组的后半部分。
     */
    public int[] exchange(int[] nums) {
        int i = 0;
        int j = nums.length - 1;
        int temp;
        while (i < j){
            while (i < j && nums[i] % 2 != 0){
                i++;
            }
            while (i < j && nums[j] % 2 == 0){
                j--;
            }
            temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }
        return nums;
    }

    /**
     * 剑指 Offer 22. 链表中倒数第k个节点
     * 输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。
     *
     * 例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。
     */
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode p = head;
        ListNode q = head;
        while (q.next != null){
            if (k < 2){
                p = p.next;
            }
            q = q.next;
            k--;
        }
        return p;
    }

    /**
     * 剑指offer 24: 定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点
     * 示例:
     *
     * 输入: 1->2->3->4->5->NULL
     * 输出: 5->4->3->2->1->NULL
     */

    /**
     * Definition for singly-linked list.
     * public class ListNode {
     * int val;
     * ListNode next;
     * ListNode(int x) { val = x; }
     * }
     */
    public ListNode reverseList(ListNode head) {
        ListNode list = new ListNode();
        list.next = null;

        ListNode p = head;

        while (p != null) {
            p = head.next;
            head.next = list.next;
            list.next = head;
            head = p;
        }
        return list.next;
    }

    /**
     * 剑指 Offer 25. 合并两个排序的链表
     * 输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode head = new ListNode();
        ListNode q = head;
        ListNode p1 = l1;
        ListNode p2 = l2;
        head.next = null;
        while (p1 != null && p2 != null){
            if (p1.val < p2.val){
                l1 = l1.next;
                p1.next = q.next;
                q.next = p1;
                q = q.next;
                p1 = l1;
            }else {
                l2 = l2.next;
                p2.next = q.next;
                q.next = p2;
                q = q.next;
                p2 = l2;
            }
        }
        if (p1 != null){
            q.next = p1;
        }
        if (p2 != null){
            q.next = p2;
        }
        return head.next;
    }

    /**
     * 剑指offer 26: 树的子结构
     * 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
     * <p>
     * B是A的子结构， 即 A中有出现和B相同的结构和节点值。
     */
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        return (A != null && B != null) && (recur(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B));
    }

    public boolean recur(TreeNode A, TreeNode B) {
        if (B == null) {
            return true;
        }
        if (A == null || A.val != B.val) {
            return false;
        }
        return recur(A.left, B.left) && recur(A.right, B.right);
    }

    /**
     * 剑指offer 27: 二叉树的镜像
     * 请完成一个函数，输入一个二叉树，该函数输出它的镜像。
     */
    public TreeNode mirrorTree(TreeNode root) {
        if (root == null || root.left == null && root.right == null) {
            return root;
        }
        Queue<TreeNode> tree = new LinkedList<>();
        TreeNode temp;
        tree.offer(root);
        while (!tree.isEmpty()) {
            TreeNode node = tree.poll();
            temp = node.left;
            node.left = node.right;
            node.right = temp;
            if (node.left != null) {
                tree.offer(node.left);
            }
            if (node.right != null) {
                tree.offer(node.right);
            }
        }
        return root;
    }

    /**
     * 剑指 Offer 29. 顺时针打印矩阵
     * 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
     */
    public int[] spiralOrder(int[][] matrix) {
        //matrix的横竖
        final int s = matrix.length;
        if (s == 0){
            return new int[0];
        }
        final int h = matrix[0].length;
        //返回函数
        int[] ans = new int[s * h];
        int n = 0;
        //标记matrix中该位是否被访问
        int[][] tag = new int[s][h];
        //方向，0向右，1向下，2向左，3向上
        int vector = 0;
        //遍历matrix序号
        int i = 0;
        int j = 0;
        while (n < ans.length) {
            switch (vector){
                case 0:
                    while (j < h && tag[i][j] == 0){
                        tag[i][j] = 1;
                        ans[n++] = matrix[i][j++];
                    }
                    break;
                case 1:
                    while (i < s && tag[i][j] == 0){
                        tag[i][j] = 1;
                        ans[n++] = matrix[i++][j];
                    }
                    break;
                case 2:
                    while (j >= 0 && tag[i][j] == 0){
                        tag[i][j] = 1;
                        ans[n++] = matrix[i][j--];
                    }
                    break;
                case 3:
                    while (s >= 0 && tag[i][j] == 0){
                        tag[i][j] = 1;
                        ans[n++] = matrix[i--][j];
                    }
                    break;
            }
            switch (vector){
                case 0:
                    j--;
                    i++;
                    break;
                case 1:
                    i--;
                    j--;
                    break;
                case 2:
                    j++;
                    i--;
                    break;
                case 3:
                    i++;
                    j++;
                    break;
            }
            vector = (vector + 1) % 4;
        }
        return ans;
    }


    /**
     * 剑指offer 30: 包含min函数的栈
     * 定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。
     */

    /**
     * 剑指 Offer 31. 栈的压入、弹出序列
     * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。
     * 假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，
     * 序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。
     */
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> putIn = new Stack<>();
        int i = 0;
        int j = 0;
        while (j < popped.length && i <= pushed.length){
            if (putIn.isEmpty() || putIn.peek() != popped[j]){
                if (i == pushed.length){
                    return false;
                }
                putIn.push(pushed[i++]);
            }else{
                putIn.pop();
                j++;
            }
        }
        return true;
    }


    /**
     * 剑指offer 32-Ⅰ: 从上到下打印二叉树
     * 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
     */
    public int[] levelOrder1(TreeNode root) {
        if (root == null) {
            return new int[0];
        }
        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> ans = new ArrayList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            ans.add(node.val);
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
        }
        int[] res = new int[ans.size()];
        int i = 0;
        for (Integer an : ans) {
            res[i++] = an;
        }
        return res;

    }

    /**
     * 剑指offer 32-Ⅱ: 从上到下打印二叉树
     * 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
     */
    public List<List<Integer>> levelOrder2(TreeNode root) {
        List<List<Integer>> ret = new ArrayList<>();
        if (root == null) {
            return ret;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> level = new ArrayList<>();
            int currentLevelCount = queue.size();
            for (int i = 0; i < currentLevelCount; i++) {
                TreeNode node = queue.poll();
                level.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            ret.add(level);
        }
        return ret;

    }

    /**
     * 剑指offer 32-Ⅲ: 从上到下打印二叉树
     * 请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推
     */
    public List<List<Integer>> levelOrder3(TreeNode root) {
        List<List<Integer>> ret = new ArrayList<>();
        if (root == null) {
            return ret;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int count = 0;
        while (!queue.isEmpty()) {
            LinkedList<Integer> level = new LinkedList<>();
            int currentLevelCount = queue.size();
            for (int i = 0; i < currentLevelCount; i++) {
                TreeNode node = queue.poll();
                if (count % 2 == 0) {
                    level.addFirst(node.val);
                } else {
                    level.addLast(node.val);
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            count++;
            ret.add(level);
        }
        return ret;
    }

    /**
     * 剑指 Offer 33. 二叉搜索树的后序遍历序列
     * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。
     */
    public boolean verifyPostorder(int[] postorder) {
        Stack<Integer> stack = new Stack<>();
        int root = Integer.MAX_VALUE;
        for(int i = postorder.length - 1; i >= 0; i--) {
            if(postorder[i] > root) return false;
            while(!stack.isEmpty() && stack.peek() > postorder[i])
                root = stack.pop();
            stack.add(postorder[i]);
        }
        return true;
    }

    /**
     * 剑指 Offer 34. 二叉树中和为某一值的路径
     *给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
     *
     * 叶子节点 是指没有子节点的节点。
     */
    List<List<Integer>> ret = new ArrayList<>();
    Deque<Integer> path = new LinkedList<>();
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        dfs(root,target);
        return ret;

    }
    private void dfs(TreeNode root, int target){
        if (root == null){
            return;
        }
        path.offerLast(root.val);
        target -= root.val;
        if (root.left == null && root.right == null && target == 0){
            ret.add(new LinkedList<>(path));
        }
        dfs(root.left, target);
        dfs(root.right, target);
        path.pollLast();


    }

    /**
     * 剑指offer 35: 复杂链表的复制
     * 请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。
     */
    Map<Node, Node> nodeValue = new HashMap<>();

    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        if (!nodeValue.containsKey(head)) {
            Node newNode = new Node(head.val);
            nodeValue.put(head, newNode);
            newNode.next = copyRandomList(head.next);
            newNode.random = copyRandomList(head.random);
        }
        return nodeValue.get(head);
    }

    /**
     * 剑指 Offer 36. 二叉搜索树与双向链表
     * 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。
     */
    public TreeNode treeToDoublyList(TreeNode root) {
        if (root == null){
            return root;
        }
        Stack<TreeNode> s = new Stack<>();
        TreeNode p = root;
        TreeNode head = new TreeNode();
        TreeNode q = head;
        while (!s.isEmpty() || p != null){
            while (p != null){
                s.push(p);
                p = p.left;
            }
            if (!s.isEmpty()){
                p = s.pop();
                p.left = q;
                q.right = p;
                q = p;
                if (p.right == null){
                    if (s.isEmpty()){
                        p.right = head.right;
                        head.right.left = p;
                        return head.right;
                    }
                    p.right = s.peek();
                    p = null;
                }else {
                    q = p;
                    p = p.right;
                }
            }
        }
        return head;
    }

    /**
     * 剑指 Offer 37. 序列化二叉树
     * 请实现两个函数，分别用来序列化和反序列化二叉树。
     *
     * 你需要设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。
     *
     * 提示：输入输出格式与 LeetCode 目前使用的方式一致，详情请参阅LeetCode 序列化二叉树的格式。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。
     */
    public String serialize(TreeNode root) {
        if (root == null){
            return "[]";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            TreeNode node = queue.poll();
            if (node != null){
                sb.append(node.val + ",");
                queue.offer(node.left);
                queue.offer(node.right);
            }else {
                sb.append("null,");
            }
        }
        sb.deleteCharAt(sb.length() - 1);
        sb.append("]");
        return sb.toString();
    }

    public TreeNode deserialize(String data) {
        if (data.equals("[]")){
            return null;
        }
        String[] arrays = data.substring(1, data.length() - 1).split(",");
        TreeNode root = new TreeNode();
        root.val = Integer.valueOf(arrays[0]);
        Queue<TreeNode> queue = new LinkedList<>();
        int i = 1;
        queue.offer(root);
        while (!queue.isEmpty()){
            TreeNode node = queue.poll();
            if (!arrays[i].equals("null")){
                node.left = new TreeNode(Integer.valueOf(arrays[i]));
                queue.offer(node.left);
            }
            i++;
            if (!arrays[i].equals("null")){
                node.right = new TreeNode(Integer.valueOf(arrays[i]));
                queue.offer(node.right);
            }
            i++;
        }
        return root;
    }

    /**
     * 剑指 Offer 38. 字符串的排列
     * 输入一个字符串，打印出该字符串中字符的所有排列。
     *
     *
     *
     * 你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。
     */
    List<String> res = new ArrayList<>();
    char[] cRes;
    public String[] permutation(String s) {
        cRes = s.toCharArray();
        dfs(0);
        return res.toArray(new String[res.size()]);
    }

    private void dfs(int x) {
        if (x == cRes.length - 1){
            res.add(String.valueOf(cRes));
            return;
        }
        Set<Character> set = new HashSet<>();
        for (int i = x; i < cRes.length; i++) {
            if (set.contains(cRes[i])){
                continue;
            }
            set.add(cRes[i]);
            swap(i, x);
            dfs(x + 1);
            swap(i, x);
        }
    }

    private void swap(int a, int b) {
        char tmp = cRes[a];
        cRes[a] = cRes[b];
        cRes[b] = tmp;
    }

    /**
     * 剑指 Offer 39. 数组中出现次数超过一半的数字
     * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
     * 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
     */
    public int majorityElement(int[] nums) {
        int temp = 0;
        int count = 0;
        for (int num : nums) {
            if (count == 0){
                temp = num;
                count++;
            }else if (num == temp){
                count++;
            }else {
                count--;
            }
        }
        return temp;
    }


    /**
     * 剑指 Offer 40. 最小的k个数
     *输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
     */
    public int[] getLeastNumbers1(int[] arr, int k) {
        List<Integer> ans = new LinkedList<>();
        int max = 0;
        for (int i : arr) {
            if (ans.size() < k){
                max = Math.max(max,i);
                ans.add(i);
            }else if (i < max){
                ans.remove((Object)max);
                ans.add(i);
                max = Collections.max(ans);
            }
        }
        int[] arr1 = ans.stream().mapToInt(Integer::valueOf).toArray();
        return arr1;
    }

    public int[] getLeastNumbers2(int[] arr, int k) {
        if (arr.length < k || k == 0){
            return new int[0];
        }
        Queue<Integer> queue = new PriorityQueue<>((v1, v2) -> v2 - v1);
        for (int i : arr) {
            if (queue.size() < k){
                queue.offer(i);
            }else if (i < queue.peek()){
                queue.poll();
                queue.offer(i);
            }
        }
        return queue.stream().mapToInt(Integer::valueOf).toArray();
    }

    public int[] getLeastNumbers3(int[] arr, int k){
        Arrays.sort(arr);
        return Arrays.copyOf(arr,k);
    }

    /**
     * 剑指 Offer 41. 数据流中的中位数
     * 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
     *
     * 例如，
     *
     * [2,3,4]的中位数是 3
     *
     * [2,3] 的中位数是 (2 + 3) / 2 = 2.5
     *
     * 设计一个支持以下两种操作的数据结构：
     *
     * void addNum(int num) - 从数据流中添加一个整数到数据结构中。
     * double findMedian() - 返回目前所有元素的中位数。
     */
    /** initialize your data structure here. */
    Queue<Integer> A, B;
    public void MedianFinder() {
        A = new PriorityQueue<>();
        B = new PriorityQueue<>((x, y) -> (y - x));
    }

    public void addNum(int num) {
        if (A.size() != B.size()){
            A.offer(num);
            B.offer(A.poll());
        }else {
            B.offer(num);
            A.offer(B.poll());
        }
    }

    public double findMedian() {
        return A.size() != B.size() ? A.peek() : (A.peek() + B.peek()) / 2.0;
    }

    /**
     * 剑指offer 42: 连续子数组的最大和
     * 输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
     * <p>
     * 要求时间复杂度为O(n)
     */
    public int maxSubArray(int[] nums) {
        int i = 0;
        int sums = nums[0];
        for (int num : nums) {
            i = Math.max(num, i + num);
            sums = Math.max(i, sums);
        }
        return sums;
    }

    /**
     * 剑指 Offer 43. 1～n 整数中 1 出现的次数
     * 输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。
     *
     * 例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。
     */
    public int countDigitOne(int n) {
        int digit = 1, res = 0;
        int high = n / 10, cur = n % 10, low = 0;
        while(high != 0 || cur != 0) {
            if(cur == 0) res += high * digit;
            else if(cur == 1) res += high * digit + low + 1;
            else res += (high + 1) * digit;
            low += cur * digit;
            cur = high % 10;
            high /= 10;
            digit *= 10;
        }
        return res;
    }

    /**
     * 剑指 Offer 44. 数字序列中某一位的数字
     * 数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。
     *
     * 请写一个函数，求任意第n位对应的数字。
     */
    public int findNthDigit(int n) {
        int digit = 1;
        long start = 1;
        long count = 9;
        while (n > count) { // 1.
            n -= count;
            digit += 1;
            start *= 10;
            count = digit * start * 9;
        }
        long num = start + (n - 1) / digit; // 2.
        return Long.toString(num).charAt((n - 1) % digit) - '0'; // 3.
    }

    /**
     * 剑指 Offer 45. 把数组排成最小的数
     * 输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
     */
    public String minNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strs[i] = String.valueOf(nums[i]);
        }
        quickStringSort(strs, 0, strs.length - 1);
        StringBuilder sb = new StringBuilder();
        for (String str : strs) {
            sb.append(str);
        }
        return sb.toString();
    }

    private void quickStringSort(String[] strs, int pre, int rear){
        if (pre < rear){
            String key = strs[pre];
            int i = pre;
            int j = rear;
            while (i < j){
                while (i < j && (strs[j] + key).compareTo(key + strs[j]) > 0){
                    j--;
                }
                if (i < j){
                    strs[i] = strs[j];
                    i++;
                }
                while (i < j && (strs[i] + key).compareTo(key + strs[i]) < 0){
                    i++;
                }
                if (i < j){
                    strs[j] = strs[i];
                    j--;
                }
            }
            strs[i] = key;
            quickStringSort(strs, pre, i - 1);
            quickStringSort(strs, i + 1, rear);
        }
    }



    /**
     * 剑指offer 46: 把数字翻译成字符串
     * 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。
     * 请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
     */
    public int translateNum(int num) {
        String src = String.valueOf(num);
        int p = 0, q = 0, r = 1;
        for (int i = 0; i < src.length(); ++i) {
            p = q;
            q = r;
            r = 0;
            r += q;
            if (i == 0) {
                continue;
            }
            String pre = src.substring(i - 1, i + 1);
            if (pre.compareTo("25") <= 0 && pre.compareTo("10") >= 0) {
                r += p;
            }
        }
        return r;


    }

    /**
     * 剑指offer 47: 礼物的最大价值
     * 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物
     */
    public int maxValue(int[][] grid) {
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (i == 0 && j == 0) {
                    continue;
                } else if (i == 0) {
                    grid[i][j] += grid[i][j - 1];
                } else if (j == 0) {
                    grid[i][j] += grid[i - 1][j];
                } else {
                    grid[i][j] += Math.max(grid[i][j - 1], grid[i - 1][j]);
                }
            }
        }
        return grid[grid.length - 1][grid[0].length - 1];
    }

    /**
     * 剑指offer 48: 最长不含重复字符的子字符串
     * 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
     */
    public int lengthOfLongestSubstring(String s) {
        int pre = 0;
        int rear = 0;
        int len = 0;
        Set<Character> flag = new HashSet<>();
        while (rear < s.length()) {
            while (flag.contains(s.charAt(rear))) {
                flag.remove(s.charAt(pre++));
            }
            len = Math.max(len, rear - pre + 1);
            flag.add(s.charAt(rear));
            rear++;
        }
        return len;
    }

    /**
     * 剑指 Offer 49. 丑数
     * 我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
     */
    public int nthUglyNumber(int n) {
        if (n <= 0){
            return -1;
        }
        int[] ans = new int[n];
        ans[0] = 1;
        int id2 = 0;
        int id3 = 0;
        int id5 = 0;
        for (int i = 1; i < n; i++) {
            ans[i] = Math.min(ans[id2] * 2, Math.min(ans[id3] * 3, ans[id5] * 5));
            if (ans[id2] * 2 == ans[i]){
                id2 += 1;
            }
            if (ans[id3] * 3 == ans[i]){
                id3 += 1;
            }
            if (ans[id5] * 5 == ans[i]){
                id5 += 1;
            }
        }
        return ans[n - 1];
    }

    /**
     * 剑指offer 50： 第一个只出现一次的字符
     * 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。
     * <p>
     * 示例 1:
     * <p>
     * 输入：s = "abaccdeff"
     * 输出：'b'
     * 示例 2:
     * <p>
     * 输入：s = ""
     * 输出：' '
     */
    public char firstUniqChar(String s) {
        Map<Character, Integer> map = new HashMap<>();
        map.put(' ', 1);
        char c = ' ';
        for (int i = 0; i < s.length(); i++) {
            c = s.charAt(i);
            if (c >= 97 && c <= 122) {
                if (!map.containsKey(c)) {
                    map.put(c, 1);
                } else {
                    map.replace(c, 2);
                }
            }
        }
        for (int i = 0; i < s.length(); i++) {
            c = s.charAt(i);
            boolean b = map.get(c) == 1;
            if (b) {
                return c;
            }
        }
        return ' ';
    }

    /**
     * 剑指 Offer 51. 数组中的逆序对
     * 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
     */
    //利用归并排序解答，在合并的时候，当左边的大于右边的，就计算逆序数
    //计算公式： mid - left + 1
    //定义一个全局的计数器变量
    int count = 0;
    public int reversePairs(int[] nums) {
        //利用归并排序解答，在合并的时候，当左边的大于右边的，就计算逆序数
        //计算公式： mid - left + 1
        //定义一个全局的计数器变量
        this.count = 0;
        mergeSort(nums, 0, nums.length - 1);
        return count;
    }

    private void mergeSort(int[] nums, int left, int right) {
        //当只有一个节点的时候，直接返回，推出递归
        if (left >= right){
            return;
        }
        int mid = (left + right) / 2;
        //左拆分
        mergeSort(nums, left, mid);
        //右拆分
        mergeSort(nums, mid + 1, right);
        //合并
        merge(nums, left, mid, right);
    }
    private void merge(int[] nums, int left, int mid, int right){
        //定义一个临时数组
        int[] temp = new int[right - left + 1];
        //定义一个指针，指向第一个数组的第一个元素
        int i = left;
        //定义一个指针，指向第二个数组的第一个元素
        int j = mid + 1;
        //定义一个指针，指向临时数组的第一个元素
        int t = 0;
        //当两个数组都有元素时，遍历比较每个元素的大小
        while (i <= mid && j <= right){
            //比较两个数组的元素，取较小的元素加入临时数组
            //并将两个指针移向下一个元素
            if (nums[i] <= nums[j]){
                temp[t++] = nums[i++];
            }else {
                //当左边数组的元素大于右边数组的元素时，就对当前元素及后面的元素的个数进行统计
                //此时这个数就是逆序数
                //用计数器记录下每次合并中的逆序数
                count += mid - i + 1;
                temp[t++]  = nums[j++];
            }
        }
        //当左边的数组没有遍历完成，直接将剩余元素加入到临时数组中
        while (i <= mid){
            temp[t++] = nums[i++];
        }
        //当右边的数组没有遍历完成，直接将剩余元素加入到临时数组中
        while (j <= right){
            temp[t++] = nums[j++];
        }
        //将新数组中的元素覆盖nums旧数组中的元素
        //此时数组的元素已经是有序的
        for (int k = 0; k < temp.length; k++) {
            nums[left + k] = temp[k];
        }
    }


    /**
     * 剑指 Offer 52. 两个链表的第一个公共节点
     * 输入两个链表，找出它们的第一个公共节点。
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int countA = 0;
        int countB = 0;
        ListNode pA = headA;
        ListNode pB = headB;
        while (pA != null){
            countA++;
            pA = pA.next;
        }
        while (pB != null){
            countB++;
            pB = pB.next;
        }
        pA = headA;
        pB = headB;
        while (pA != null){
            if (countA > countB){
                pA = pA.next;
                countA--;
            }else if (countA < countB){
                pB = pB.next;
                countB--;
            }else {
                if (pA == pB){
                    return pA;
                }else {
                    pA = pA.next;
                    pB = pB.next;
                }
            }
        }
        return null;
    }


    /**
     * 剑指offer 53-Ⅰ: 在排序数组中出现的次数
     * 统计一个数字在排序数组中出现的次数。
     * <p>
     * 示例 1:
     * <p>
     * 输入: nums = [5,7,7,8,8,10], target = 8
     * 输出: 2
     * 示例 2:
     * <p>
     * 输入: nums = [5,7,7,8,8,10], target = 6
     * 输出: 0
     */
    public int search(int[] nums, int target) {
        if (nums.length == 0) {
            return 0;
        }
        int low = 0;
        int high = nums.length - 1;
        int mid;

        while (low <= high) {
            mid = (low + high) / 2;
            if (nums[mid] == target) {
                int count = 1;
                int i = 1;
                while (mid + i < nums.length && nums[mid + i] == target || mid - i > -1 && nums[mid - i] == target) {
                    if (mid + i < nums.length && nums[mid + i] == target) {
                        count++;
                    }
                    if (mid - i > -1 && nums[mid - i] == target) {
                        count++;
                    }
                    i++;
                }
                return count;
            } else if (nums[mid] > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return 0;
    }

    /**
     * 剑指offer 53-Ⅱ: 0~n-1中缺失的数字
     * 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
     */
    public int missingNumber(int[] nums) {
        if (nums[0] != 0) {
            return 0;
        }
        int low = 0;
        int high = nums.length - 1;
        int mid = 0;
        while (low <= high) {
            mid = (low + high) / 2;
            if (nums[mid] == mid) {
                low = mid + 1;
            } else if (nums[mid] == mid + 1 && nums[mid - 1] != mid) {
                return mid;

            } else {
                high = mid - 1;
            }
        }

        return mid + 1;
    }

    /**
     * 剑指 Offer 54. 二叉搜索树的第k大节点
     * 给定一棵二叉搜索树，请找出其中第 k 大的节点的值。
     */
    public int kthLargest(TreeNode root, int k) {
        if (root == null){
            return -1;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null){
            while (p != null){
                stack.push(p);
                p = p.right;
            }
            if (k == 1){
                return stack.pop().val;
            }else {
                k--;
                p = stack.pop();
                p = p.left;
            }
        }
        return -1;
    }


    /**
     * 剑指 Offer 55 - I. 二叉树的深度
     * 输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。
     */
    public int maxDepth(TreeNode root) {
        if (root == null){
            return 0;
        }else {
            int leftDeep = maxDepth(root.left);
            int rightDeep = maxDepth(root.right);
            return Math.max(leftDeep, rightDeep) + 1;
        }
    }

    /**
     * 剑指 Offer 55 - II. 平衡二叉树
     * 输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。
     */
    public boolean isBalanced(TreeNode root) {
        return height(root) >= 0;
    }
    private int height(TreeNode root){
        if (root == null){
            return 0;
        }
        int leftDeep = height(root.left);
        int rightDeep = height(root.right);
        if (leftDeep == -1 || rightDeep == -1 || Math.abs(leftDeep - rightDeep) > 1){
            return -1;
        }else {
            return Math.max(leftDeep, rightDeep) + 1;
        }
    }

    /**
     * 剑指 Offer 56 - I. 数组中数字出现的次数
     * 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
     */
    public int[] singleNumbers(int[] nums) {
        int ans = 0;
        for (int num : nums) {
            ans ^= num;
        }
        int flag = ans & -ans;
        int res = 0;
        for (int num : nums) {
            if ((flag & num) != 0){
                res ^= num;
            }
        }
        return new int[]{res, ans ^ res};
    }

    /**
     * 剑指 Offer 56 - II. 数组中数字出现的次数 II
     * 在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。
     */
    public int singleNumber2(int[] nums) {
        int ones = 0;
        int twos = 0;
        for (int num : nums) {
            ones = ones ^ num & ~twos;
            twos = twos ^ num & ~ones;
        }
        return ones;
    }


    /**
     * 剑指 Offer 57. 和为s的两个数字
     * 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
     */
    public int[] twoSum57(int[] nums, int target) {
        int i = 0;
        int j = nums.length - 1;
        while (i < j){
            if (nums[i] + nums[j] == target){
                int[] ans = {nums[i], nums[j]};
                return ans;
            }else if (nums[i] + nums[j] > target){
                j--;
            }else {
                i++;
            }
        }
        return new int[0];
    }

    /**
     * 剑指 Offer 57 - II. 和为s的连续正数序列
     * 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
     *
     * 序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
     */
    public int[][] findContinuousSequence(int target) {
        List<int[]> lists = new ArrayList<>();
        int i = 1, j = 1;
        int sum = 0;
        while (i <= target / 2){
            if (sum < target){
                sum += j;
                j++;
            }else if (sum > target){
                sum -= i;
                i++;
            }else {
                int[] arr = new int[j - i];
                for (int k = i; k < j; k++) {
                    arr[k - i] = k;
                }
                lists.add(arr);
                sum -= i;
                i++;
            }
        }
        return lists.toArray(new int[lists.size()][]);
    }


    /**
     * 剑指 Offer 58 - I. 翻转单词顺序
     * 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。
     */
    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        Stack<String> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c != ' '){
                sb.append(c);
            }else if (sb.length() > 0){
                String m = sb.toString();
                stack.push(m);
                sb.delete(0,sb.length());
            }
        }
        while (!stack.isEmpty()){
            if (!sb.isEmpty()){
                sb.append(' ');
            }
            sb.append(stack.pop());
        }
        return sb.toString();
    }

    public String reverseWords2(String s){
        s = s.trim();
        List<String> list = Arrays.asList(s.split("\\s+"));
        Collections.reverse(list);
        return String.join(" ", list);
    }

    /**
     * 剑指offer 58-Ⅱ: 左旋转字符串
     */
    public String reverseLeftWords(String s, int n) {
        StringBuffer bfl = new StringBuffer(s.substring(0, n));
        StringBuffer bfr = new StringBuffer(s.substring(n));
        bfl.reverse();
        bfr.reverse();
        bfl.append(bfr);
        bfl.reverse();
        return bfl.toString();


    }

    public String reverseLeftWords2(String s, int n) {
        char[] chars = s.toCharArray();
        n = chars.length - n;
        reverse(chars, 0, chars.length - 1);
        reverse(chars, 0, n - 1);
        reverse(chars, n, chars.length - 1);
        return new String(chars);
    }

    private void reverse(char[] chars, int low, int high){
        while (low < high){
            chars[low] ^= chars[high];
            chars[high] ^= chars[low];
            chars[low++] ^= chars[high--];
        }
    }

    /**
     * 剑指 Offer 59 - I. 滑动窗口的最大值
     * 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums.length == 0 || k == 0) {
            return new int[0];
        }
        Deque<Integer> deque = new LinkedList<>();
        int[] ans = new int[nums.length - k + 1];
        for(int j = 0, i = 1 - k; j < nums.length; i++, j++){
            //删除Deque中对应的nums[i - 1]
            if (i > 0 && deque.peekFirst() == nums[i - 1]){
                deque.removeFirst();
            }
            //保持Deque递减
            while (!deque.isEmpty() && deque.peekLast() < nums[j]){
                deque.removeLast();
            }
            deque.addLast(nums[j]);
            //记录窗口最大值
            if (i >= 0){
                ans[i] = deque.peekFirst();
            }
        }
        return ans;
    }

    /**
     * 剑指 Offer 59 - II. 队列的最大值
     * 请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。
     *
     * 若队列为空，pop_front 和 max_value需要返回 -1
     */
    Queue<Integer> queueMax;
    Deque<Integer> dequeMax;
    public void MaxQueue() {
        queueMax = new LinkedList<>();
        dequeMax = new LinkedList<>();
    }

    public int max_value() {
        if (queueMax.isEmpty()){
            return -1;
        }
        return dequeMax.peekFirst();
    }

    public void push_back(int value) {
        while (!dequeMax.isEmpty() && dequeMax.peekLast() < value){
            dequeMax.pollLast();
        }
        dequeMax.offerLast(value);
        queueMax.offer(value);

    }

    public int pop_front() {
        if (queueMax.isEmpty()){
            return -1;
        }
        int ans = queueMax.poll();
        if (ans == dequeMax.peekFirst()){
            dequeMax.pop();
        }
        return ans;
    }

    /**
     * 剑指 Offer 60. n个骰子的点数
     * 把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
     *
     * 你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。
     */
    public double[] dicesProbability(int n) {
        double[] dp = new double[6];
        Arrays.fill(dp, 1.0 / 6.0);
        for (int i = 2; i <= n; i++) {
            double[] tmp = new double[5 * i + 1];
            for (int j = 0; j < dp.length; j++) {
                for (int k = 0; k < 6; k++) {
                    tmp[j + k] += dp[j] / 6.0;
                }
            }
            dp = tmp;
        }
        return dp;
    }

    /**
     * 剑指 Offer 61. 扑克牌中的顺子
     * 从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。
     */
    public boolean isStraight(int[] nums) {
        Set<Integer> set = new HashSet<>();
        int max = 0;
        int min = 14;
        for (int num : nums) {
            if (num == 0){
                continue;
            }
            if (set.contains(num)){
                return false;
            }
            set.add(num);
            max = Math.max(num, max);
            min = Math.min(num, min);

        }
        return max - min < 5;
    }

    /**
     * 剑指 Offer 62. 圆圈中最后剩下的数字
     * 0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。
     *
     * 例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。
     */
    public int lastRemaining(int n, int m) {
        int f = 0;
        for (int i = 2; i != n + 1; ++i) {
            f = (m + f) % i;
        }
        return f;
    }



    /**
     * 剑指offer 63: 股票的最大利润
     * 假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？
     */
    public int maxProfit(int[] prices) {
        int cost = Integer.MAX_VALUE;
        int profit = 0;
        for (int price : prices) {
            cost = Math.min(cost, price);
            profit = Math.max(profit, price - cost);
        }
        return profit;
    }


    /**
     * 剑指 Offer 64. 求1+2+…+n
     * 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）
     */
    public int sumNums(int n) {
        return (1 + n) * n / 2;
    }

    /**
     * 剑指 Offer 65. 不用加减乘除做加法
     * 写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。
     */
    public int add(int a, int b) {
        if (b == 0) {
            return a;
        }

        // 转换成非进位和 + 进位
        return add(a ^ b, (a & b) << 1);
    }

    /**
     * 剑指 Offer 66. 构建乘积数组
     * 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积, 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
     */
    public int[] constructArr(int[] a) {
        int length = a.length;
        if(length == 0){
            return new int[0];
        }
        // L 和 R 分别表示左右两侧的乘积列表
        int[] L = new int[length];
        int[] R = new int[length];
        int[] answer = new int[length];
        // L[i] 为索引 i 左侧所有元素的乘积
        // 对于索引为 '0' 的元素，因为左侧没有元素，所以 L[0] = 1
        L[0] = 1;
        for (int i = 1; i < length; i++) {
            L[i] = a[i - 1] * L[i - 1];
        }
        // R[i] 为索引 i 右侧所有元素的乘积
        // 对于索引为 'length-1' 的元素，因为右侧没有元素，所以 R[length-1] = 1
        R[length - 1] = 1;
        for (int i = length - 2; i >= 0; i--) {
            R[i] = a[i + 1] * R[i + 1];
        }
        // 对于索引 i，除 a[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
        for (int i = 0; i < length; i++) {
            answer[i] = L[i] * R[i];
        }
        return answer;
    }

    /**
     * 剑指 Offer 67. 把字符串转换成整数
     * 写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。
     *
     *
     * 首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
     *
     * 当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
     *
     * 该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。
     *
     * 注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。
     *
     * 在任何情况下，若函数不能进行有效的转换时，请返回 0。
     *
     * 说明：
     *
     * 假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为[−231, 231− 1]。如果数值超过这个范围，请返回 INT_MAX (231− 1) 或INT_MIN (−231) 。
     */
    public int strToInt(String str) {
        char[] c = str.trim().toCharArray();
        if(c.length == 0) return 0;
        int res = 0, bndry = Integer.MAX_VALUE / 10;
        int i = 1, sign = 1;
        if(c[0] == '-') sign = -1;
        else if(c[0] != '+') i = 0;
        for(int j = i; j < c.length; j++) {
            if(c[j] < '0' || c[j] > '9') break;
            if(res > bndry || res == bndry && c[j] > '7') return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            res = res * 10 + (c[j] - '0');
        }
        return sign * res;
    }


    /**
     * 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
     * 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
     *
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
     *
     * 例如，给定如下二叉搜索树: root =[6,2,8,0,4,7,9,null,null,3,5]
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        if (root.val < p.val && root.val < q.val) return lowestCommonAncestor(root.right, p, q);
        else if (root.val > p.val && root.val > q.val) return lowestCommonAncestor(root.left, p, q);
        else return root;
    }

    /**
     * 剑指 Offer 68 - II. 二叉树的最近公共祖先
     * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
     *
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
     */
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        List<TreeNode> path1 = new LinkedList<>();
        List<TreeNode> path2 = new LinkedList<>();
        getTree(root,p,path1);
        getTree(root,q,path2);
        TreeNode ans = null;
        int len = Math.min(path1.size(), path2.size());
        for (int i = 0; i < len; i++) {
            if (path1.get(i).equals(path2.get(i)))
                ans = path1.get(i);
        }
        return ans;
    }
    private void getTree(TreeNode root, TreeNode node, List<TreeNode> path){
        if (root == null)
            return;
        path.add(root);
        if (root == node)
            return;
        if (path.get(path.size() - 1) != node)
            getTree(root.left, node, path);
        if (path.get(path.size() - 1) != node)
            getTree(root.right, node, path);
        if (path.get(path.size() - 1) != node)
            path.remove(path.size() - 1);
    }

    public TreeNode lowestCommonAncestor3(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == root || q == root) {
            return root;
        }

        TreeNode l = lowestCommonAncestor(root.left, p, q);
        TreeNode r = lowestCommonAncestor(root.right, p, q);

        return l == null ? r : (r == null ? l : root);
    }


    /**
     * 1: 两数之和
     * 给定一个整数数组 nums和一个整数目标值 target，请你在该数组中找出 和为目标值 target 的那两个整数，并返回它们的数组下标。
     * <p>
     * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
     * <p>
     * 你可以按任意顺序返回答案。
     */
    public int[] twoSum1(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{map.get(target - nums[i]), i};
            } else {
                map.put(nums[i], i);
            }
        }
        return new int[0];
    }

    public int[] twoSum2(int[] nums, int target) {
        int[] copy = Arrays.copyOf(nums, nums.length);
        Arrays.sort(nums);
        int i = 0;
        int j = nums.length - 1;
        while (i < j) {
            int sum = nums[i] + nums[j];
            if (sum == target) {
                int pre = Arrays.binarySearch(copy, nums[i]);
                int rear = Arrays.binarySearch(copy, nums[j]);
                return new int[]{pre, rear};
            } else if (sum > target) {
                j--;
            } else {
                j++;
            }
        }
        return new int[]{};
    }

    //1ms
    public int[] twoSum3(int[] nums, int target) {

        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])){
                return new int[]{i, map.get(target - nums[i])};
            }else {
                map.put(nums[i], i);
            }
        }
        return new int[2];
    }


    /**
     * 2. 两数相加
     * 给你两个非空 的链表，表示两个非负的整数。它们每位数字都是按照逆序的方式存储的，并且每个节点只能存储一位数字。
     *
     * 请你将两个数相加，并以相同形式返回一个表示和的链表。
     *
     * 你可以假设除了数字 0 之外，这两个数都不会以 0开头。
     */
    int plusTag = 0;
    ListNode p;
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode ans = new ListNode();
        p = ans;
        ans.next = null;
        while (l1 != null && l2 != null){
            int sum = l1.val + l2.val + plusTag;
            addTwoNumbersHelper(sum);
            l1 = l1.next;
            l2 = l2.next;
        }
        while (l1 != null){
            int sum = l1.val + plusTag;
            addTwoNumbersHelper(sum);
            l1 = l1.next;
        }
        while (l2 != null){
            int sum = l2.val + plusTag;
            addTwoNumbersHelper(sum);
            l2 = l2.next;
        }
        if (plusTag == 1){
            ListNode node = new ListNode();
            node.val = plusTag;
            node.next = null;
            p.next = node;
        }
        return ans.next;
    }
    private void addTwoNumbersHelper(int sum){
        ListNode node = new ListNode();
        node.val = sum % 10;
        plusTag = sum / 10;
        node.next = null;
        p.next = node;
        p = p.next;
    }

    /**
     * 3. 无重复字符的最长子串
     * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
     */
    public int lengthOfLongestSubstring3(String s) {
        char pre = 0;
        char rear = 0;
        int len = 0;
        Set<Character> set = new HashSet<>();
        while (rear < s.length()){
            while (set.contains(s.charAt(rear))){
                set.remove(s.charAt(pre++));
            }
            len = Math.max(len, rear - pre + 1);
            set.add(s.charAt(rear++));
        }
        return len;
    }

    public int lengthOfLongestSubstring_3(String s) {
        char[] chars = s.toCharArray();
        Set<Character> set = new HashSet<>();
        int maxLen = Math.min(s.length(), 1);
        int low = 0;
        for (int high = 0; high < chars.length; high++){
            if (!set.add(chars[high])){
                maxLen = Math.max(maxLen, high - low);
                while (low < high && chars[low] != chars[high]){
                    set.remove(chars[low++]);
                }
                low++;
            }
        }
        return Math.max(maxLen, chars.length - low);
    }


    /**
     * 5. 最长回文子串
     * 给你一个字符串 s，找到 s 中最长的回文子串。
     */
    public String longestPalindrome(String s) {
        if (s.length() < 2){
            return s;
        }
        int maxLen = 1;
        char[] chars = s.toCharArray();
        StringBuilder ans = new StringBuilder();
        ans.append(chars[0]);
        for (int k = 0; k <= 1; k++) {
            for (int i = 1; i <= s.length() - 1; i++) {
                int l = 1;
                StringBuilder sb = new StringBuilder();
                while (i - l >= 0 && i + l - k < s.length() && chars[i - l] == chars[i + l - k]) {
                    sb.append(chars[i + l - k]);
                    l++;
                }
                int t = k == 0 ? 1 : 0;
                if (2 * (l - 1) + t > maxLen) {
                    maxLen = 2 * (l - 1) + t;
                    ans = new StringBuilder();
                    ans.append(sb.reverse()).append(k == 0 ? String.valueOf(chars[i]) : "").append(sb.reverse());
                }
            }
        }
        return ans.toString();
    }

    public String longestPalindrome2(String s) {
        char[] chars = s.toCharArray();
        int len, index = 0, max = 0, twin = 0;
        for (int j = 0; j <= 1; j++) {
            for (int i = 0; i < chars.length - 1; i++) {
                len = 0;
                while (i - len + j >= 1 && i + len + 1 < chars.length && chars[i - len - 1 + j] == chars[i + len + 1]){
                    len++;
                }
                if (len > max){
                    max = len;
                    index = i;
                    twin = j;
                }
            }
        }
        return s.substring(index - max + twin, index + max + 1);
    }

    /**
     * 6. Z 字形变换
     * 将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
     */
    public String convert6_1(String s, int numRows) {
        if (numRows < 2){
            return s;
        }
        char[] chars = s.toCharArray();
        char[][] strings = new char[numRows][chars.length];
        int i = 0;
        int j = 0;
        boolean flag = true;
        int h = numRows - 2;
        for (char a : chars) {
            if (i < numRows && flag){
                strings[i++][j] = a;
            }else if (i > 0 && h > 0 && !flag){
                strings[i--][j++] = a;
                h--;
            }
            if (i == numRows && flag){
                j++;
                if (numRows == 2){
                    i = 0;
                    continue;
                }
                i = numRows - 2;
                flag = false;
            }else if (i == 0 && !flag){
                flag = true;
                h = numRows - 2;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (char[] s1 : strings) {
            for (char c : s1) {
                if (c != 0){
                    sb.append(c);
                }
            }
        }
        return sb.toString();
    }
    public String convert6_2(String s, int numRows) {
        if (numRows < 2){
            return s;
        }
        List<StringBuilder> list = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            list.add(new StringBuilder());
        }
        int i = 0;
        int flag = -1;
        for (char c : s.toCharArray()) {
            list.get(i).append(c);
            if (i == 0 || i == numRows - 1){
                flag = -flag;
            }
            i += flag;
        }
        StringBuilder sb = new StringBuilder();
        for (StringBuilder res : list) {
            sb.append(res);
        }
        return sb.toString();
    }

    /**
     * 7. 整数反转
     * 给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。
     *
     * 如果反转后整数超过 32 位的有符号整数的范围[−231, 231− 1] ，就返回 0。
     *
     * 假设环境不允许存储 64 位整数（有符号或无符号）。
     */
    public int reverse(int x) {
        int res = 0;
        while (x != 0){
            int temp = x % 10;
            if (res > 214748364 || res == 214748364 && temp > 7 || res < -214748364 || res == -214748364 && temp < -8){
                return 0;
            }
            res = res * 10 + temp;
            x /= 10;
        }
        return res;
    }



    /**
     * 8. 字符串转换整数 (atoi)
     * 请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。
     */
    public int myAtoi(String s) {
        char[] chars = s.toCharArray();
        int index = 0;
        while (index < chars.length && chars[index] == ' '){
            index++;
        }
        if (index == chars.length){
            return 0;
        }
        int flag = 1;
        if (chars[index] == '-'){
            flag = -1;
            index++;
        }else if (chars[index] == '+'){
            index++;
        }
        int res = 0, last = 0; // last记录上一个记录，用来判断是否溢出
        while (index < chars.length){
            char curChar = chars[index];
            if (curChar < '0' || curChar > '9'){
                break;
            }
            last = res;
            res = res * 10 + curChar - '0';
            if (last != res / 10){
                return flag == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            index++;
        }
        return res * flag;
    }

    /**
     * 9. 回文数
     * 给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。
     *
     * 回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
     *
     * 例如，121 是回文，而 123 不是。
     */
    public boolean isPalindrome(int x) {
        String s = String.valueOf(x);
        for (int i = 0; i < s.length() / 2; i++) {
            if (s.charAt(i) != s.charAt(s.length() - 1 - i)){
                return false;
            }
        }
        return true;
    }


    /**
     * 11： 盛最多水的容器
     * 给定一个长度为 n 的整数数组height。有n条垂线，第 i 条线的两个端点是(i, 0)和(i, height[i])。
     * <p>
     * 找出其中的两条线，使得它们与x轴共同构成的容器可以容纳最多的水。
     * <p>
     * 返回容器可以储存的最大水量。
     * <p>
     * 说明：你不能倾斜容器。
     */
    public int maxArea(int[] height) {
        int area = 0;
        int i = 0;
        int j = height.length - 1;

        int wallLeft;
        int wallRight;
        while (i < j) {
            wallLeft = height[i];
            wallRight = height[j];
            area = Math.max(area, (j - i) * Math.min(wallLeft, wallRight));
            if (wallLeft > wallRight) {
                j--;
            } else {
                i++;
            }
        }
        return area;
    }

    public int maxArea2(int[] height) {
        int res = 0;
        int left = 0;
        int right = height.length - 1;
        while (left < right){
            int higher;
            if (height[left] < height[right]){
                higher = height[left];
                left++;
            }else {
                higher = height[right];
                right--;
            }
            res = Math.max(res, (right - left + 1) * higher);
        }
        return res;
    }

    /**
     * 12. 整数转罗马数字
     * 罗马数字包含以下七种字符：I，V，X，L，C，D和M。
     *
     * 字符          数值
     * I             1
     * V             5
     * X             10
     * L             50
     * C             100
     * D             500
     * M             1000
     * 例如， 罗马数字 2 写做II，即为两个并列的 1。12 写做XII，即为X+II。 27 写做XXVII, 即为XX+V+II。
     *
     * 通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做IIII，而是IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为IX。这个特殊的规则只适用于以下六种情况：
     *
     * I可以放在V(5) 和X(10) 的左边，来表示 4 和 9。
     * X可以放在L(50) 和C(100) 的左边，来表示 40 和90。
     * C可以放在D(500) 和M(1000) 的左边，来表示400 和900。
     * 给你一个整数，将其转为罗马数字。
     */
    public String intToRoman(int num) {
        int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] rom = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            while (num >= values[i]){
                sb.append(rom[i]);
                num -= values[i];
            }
        }
        return sb.toString();
    }

    /**
     * 13. 罗马数字转整数
     */
    public int romanToInt(String s) {
        Map<Character,Integer> map = new HashMap<>();
        map.put('I',1);
        map.put('V',5);
        map.put('X',10);
        map.put('L',50);
        map.put('C',100);
        map.put('D',500);
        map.put('M',1000);
        int ans = 0;
        for (int i = 0; i < s.length(); i++) {
            int p = map.get(s.charAt(i));
            if (i < s.length() - 1 && p < map.get(s.charAt(i + 1))){
                ans -= p;
            }else {
                ans += p;
            }
        }
        return ans;

    }

    /**
     * 14. 最长公共前缀
     * 编写一个函数来查找字符串数组中的最长公共前缀。
     *
     * 如果不存在公共前缀，返回空字符串 ""。
     */
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0){
            return "";
        }
        //字符串位
        int i = 0;
        boolean flag = true;
        while (i < strs[0].length() && flag){
            char temp = strs[0].charAt(i);
            for (String str : strs) {
                if (str.length() < i + 1 || str.charAt(i) != temp){
                    flag = false;
                    i--;
                    break;
                }
            }
            i++;
        }
        return i > 0 ? strs[0].substring(0, i) : "";
    }

    /**
     * 15：三数之和
     * 给你一个包含 n 个整数的数组nums，判断nums中是否存在三个元素 a，b，c ，使得a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> threesum = new ArrayList<>();
        if (nums.length < 3) {
            return threesum;
        }
        Arrays.sort(nums);

        Set<List<Integer>> sums = new HashSet<>();
        for (int i = 0; i < nums.length - 2; i++) {
            List<List<Integer>> lists = twoSum(Arrays.copyOfRange(nums, i + 1, nums.length), -nums[i]);
            for (List<Integer> list : lists) {
                list.add(nums[i]);
                list.sort(((o1, o2) -> o1 - o2));
                sums.add(list);
            }

        }
        threesum = new ArrayList<>(sums);
        return threesum;
    }

    public List<List<Integer>> twoSum(int[] nums, int target) {
        List<List<Integer>> sum = new ArrayList<>();
        int i = 0;
        int j = nums.length - 1;
        while (i < j) {
            if (nums[i] + nums[j] == target) {
                List<Integer> list = new ArrayList<>();
                list.add(nums[i]);
                list.add(nums[j]);
                sum.add(list);
                if (target == 0) {
                    break;
                } else if (target < 0) {
                    j--;
                } else {
                    i++;
                }
            } else if (nums[i] + nums[j] < target) {
                i++;
            } else {
                j--;
            }
        }
        return sum;
    }

    //回溯法，可行但超时
    public List<List<Integer>> threeSum2(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length < 3){
            return res;
        }
        Arrays.sort(nums);
        backtrack(nums, res, new ArrayList<Integer>(), 0, 0, 3 , 0);
        return res;
    }

    private void backtrack(int[] nums, List<List<Integer>> res, ArrayList<Integer> curPath, int curSum, int index, int len, int key) {
        if (curPath.size() == len){
            if (curSum == key){
                res.add(new ArrayList<>(curPath));
            }
            return;
        }
        for (int i = index; i < nums.length; i++) {
            if (i > index && nums[i] == nums[i - 1] || curSum + nums[i] > key){
                continue;
            }
            curPath.add(nums[i]);
            curSum += nums[i];
            backtrack(nums, res, curPath, curSum, i + 1, len, key);
            curPath.remove(curPath.size() - 1);
            curSum -= nums[i];
        }
    }

    public List<List<Integer>> threeSum3(int[] nums) {
        Set<List<Integer>> set = new HashSet<>();
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length < 3){
            return res;
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0){
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]){
                continue;
            }
            twoSumForThree(set, nums, -nums[i], i + 1);
        }
        for (List<Integer> list : set) {
            res.add(list);
        }
        return res;
    }

    private void twoSumForThree(Set<List<Integer>> res, int[] nums, int target, int index) {
        int low = index, high = nums.length - 1;
        while (low < high){
            if (nums[low] + nums[high] == target){
                List<Integer> list = new ArrayList<>();
                list.add(nums[low++]);
                list.add(nums[high--]);
                list.add(-target);
                res.add(list);
            }else if (nums[low] + nums[high] > target){
                high--;
            }else {
                low++;
            }
        }
    }

    public List<List<Integer>> threeSum4(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length < 3){
            return res;
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0){
                return res;
            }
            if (i > 0 && nums[i] == nums[i - 1]){
                continue;
            }
            int low = i + 1;
            int high = nums.length - 1;
            while (low < high){
                int temp = nums[i] + nums[low] + nums[high];
                if (temp == 0){
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[i]);
                    list.add(nums[low]);
                    list.add(nums[high]);
                    res.add(list);
                    while (low < high && nums[low] == nums[low + 1]){
                        low++;
                    }
                    while (low < high && nums[high] == nums[high - 1]){
                        high--;
                    }
                    low++;
                    high--;
                }else if (temp < 0){
                    low++;
                }else {
                    high--;
                }
            }
        }
        return res;
    }

    /**
     * 16. 最接近的三数之和
     * 给你一个长度为 n 的整数数组nums和 一个目标值target。请你从 nums 中选出三个整数，使它们的和与target最接近。
     *
     * 返回这三个数的和。
     *
     * 假定每组输入只存在恰好一个解。
     */
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int best = 1000000, sum = 1000000, last = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]){
                continue;
            }
            int j = i + 1;
            int k = nums.length - 1;
            while (j < k){
                last = sum;
                sum =  nums[i] + nums[j] + nums[k];
                if (sum == target){
                    return sum;
                }
                if (sum == last){
                    if (sum > target){
                        k--;
                    }else {
                        j++;
                    }
                    continue;
                }
                if (Math.abs(sum - target) < Math.abs(best - target)){
                    best = sum;
                }
                if (sum > target){
                    k--;
                }else {
                    j++;
                }
            }
        }
        return best;
    }


    /**
     * 17. 电话号码的字母组合
     *给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
     */
    Map<Character, String[]> letterMap;
    public List<String> letterCombinations(String digits) {
        List<String> combinations = new LinkedList<>();
        if (digits.length() == 0){
            return combinations;
        }
        initialLetter();
        char[] letterChars = digits.toCharArray();
        backtrack(letterChars, combinations, 0, new StringBuffer());
        return combinations;


    }

    private void backtrack(char[] chars, List<String> combinations, int index, StringBuffer combination) {
        if (index == chars.length){
            combinations.add(combination.toString());
        }else {
            String[] strs = letterMap.get(chars[index]);
            for (int i = 0; i < strs.length; i++) {
                combination.append(strs[i]);
                backtrack(chars, combinations, index + 1, combination);
                combination.deleteCharAt(index);
            }
        }
    }

    private void initialLetter(){
        letterMap = new HashMap<>();
        letterMap.put('2', new String[]{"a", "b", "c"});
        letterMap.put('3', new String[]{"d", "e", "f"});
        letterMap.put('4', new String[]{"g", "h", "i"});
        letterMap.put('5', new String[]{"j", "k", "l"});
        letterMap.put('6', new String[]{"m", "n", "o"});
        letterMap.put('7', new String[]{"p", "q", "r", "s"});
        letterMap.put('8', new String[]{"t", "u", "v"});
        letterMap.put('9', new String[]{"w", "x", "y", "z"});
    }

    public List<String> letterCombinations2(String digits) {
        List<String> res = new ArrayList<>();
        if (digits == null || digits.length() == 0){
            return res;
        }
        Map<Integer, String[]> map = new HashMap<>();
        map.put(2, new String[]{"a", "b", "c"});
        map.put(3, new String[]{"d", "e", "f"});
        map.put(4, new String[]{"g", "h", "i"});
        map.put(5, new String[]{"j", "k", "l"});
        map.put(6, new String[]{"m", "n", "o"});
        map.put(7, new String[]{"p", "q", "r", "s"});
        map.put(8, new String[]{"t", "u", "v"});
        map.put(9, new String[]{"w", "x", "y", "z"});
        backtrack(res, map, digits, new StringBuilder(), 0);
        return res;
    }

    private void backtrack(List<String> res, Map<Integer, String[]> map, String digits, StringBuilder cur, int index) {
        if (cur.length() == digits.length()){
            res.add(cur.toString());
            return;
        }
        String[] strings = map.get(digits.charAt(index) - '0');
        for (int i = 0; i < strings.length; i++) {
            cur.append(strings[i]);
            backtrack(res, map, digits, cur, index + 1);
            cur.deleteCharAt(index);
        }
    }

    /**
     * 18. 四数之和
     * 给你一个由 n 个整数组成的数组nums ，和一个目标值 target 。
     * 请你找出并返回满足下述全部条件且不重复的四元组[nums[a], nums[b], nums[c], nums[d]]（若两个四元组元素一一对应，则认为两个四元组重复）：
     *
     * 0 <= a, b, c, d< n
     * a、b、c 和 d 互不相同
     * nums[a] + nums[b] + nums[c] + nums[d] == target
     * 你可以按 任意顺序 返回答案 。
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        if (nums.length < 4){
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        int end = nums.length - 3;
        for (int i = nums.length - 4; i >= 0; i--) {
            if (nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] <= target){
                end = i;
                break;
            }
        }
        Set<List<Integer>> res = new HashSet<>();
        for (int i = 0; i <= end; i++) {
            List<List<Integer>> lists = threeSumForFour(Arrays.copyOfRange(nums, i + 1, nums.length), -nums[i] + target);
            for (List<Integer> list : lists) {
                list.add(nums[i]);
                res.add(list);
            }
        }
        return new ArrayList<>(res);
    }

    public List<List<Integer>> threeSumForFour(int[] nums, int target) {
        if (nums.length < 3) {
            return new ArrayList<>();
        }
        int end = nums.length - 2;
        for (int i = nums.length - 3; i >= 0; i--) {
            if (nums[i] + nums[i + 1] + nums[i + 2] <= target){
                end = i;
                break;
            }
        }
        Set<List<Integer>> res = new HashSet<>();
        for (int i = 0; i <= end; i++) {
            List<List<Integer>> lists = twoSumForFour(Arrays.copyOfRange(nums, i + 1, nums.length), -nums[i] + target);
            for (List<Integer> list : lists) {
                list.add(nums[i]);
                res.add(list);
            }

        }
        return new ArrayList<>(res);
    }

    public List<List<Integer>> twoSumForFour(int[] nums, int target) {
        List<List<Integer>> sum = new ArrayList<>();
        int i = 0;
        int j = nums.length - 1;
        while (i < j) {
            if (nums[i] + nums[j] == target) {
                List<Integer> list = new ArrayList<>();
                list.add(nums[i]);
                list.add(nums[j]);
                sum.add(list);
                i++;
            } else if (nums[i] + nums[j] < target) {
                i++;
            } else {
                j--;
            }
        }
        return sum;
    }

    public List<List<Integer>> fourSum2(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]){
                continue;
            }
            for (int j = i + 1; j < nums.length; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]){
                    continue;
                }
                int left = j + 1;
                int right = nums.length - 1;
                while (left < right){
                    int sum = nums[i] + nums[j];
                    int temp = nums[left] + nums[right];
                    int check = check(sum, temp);
                    sum += temp;
                    if (check == 1 || sum > target){
                        right--;
                    }else if (check == -1 || sum < target){
                        left++;
                    }else {
                        res.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]){
                            left++;
                        }
                        while (left < right && nums[right] == nums[right - 1]){
                            right--;
                        }
                        left++;
                        right--;
                    }
                }
            }
        }
        return res;
    }

    //判断整数相加是否溢出
    private int check(int i, int j){
        if (i > 0 && j > 0 &&  i > Integer.MAX_VALUE - j){
            return 1;
        }
        if (i < 0 && j < 0 && i < Integer.MIN_VALUE - j){
            return -1;
        }
        return 0;
    }

    /**
     * 19. 删除链表的倒数第 N 个结点
     * 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode l = new ListNode();
        l.next = head;
        ListNode pre = l;
        ListNode rear = l;
        for (int i = 0; i < n; i++) {
            rear = rear.next;
        }
        while (rear.next != null){
            pre = pre.next;
            rear = rear.next;
        }
        pre.next = pre.next.next;
        return l.next;
    }

    /**
     * 20. 有效的括号
     * 给定一个只包括 '('，')'，'{'，'}'，'['，']'的字符串 s ，判断字符串是否有效。
     *
     * 有效字符串需满足：
     *
     * 左括号必须用相同类型的右括号闭合。
     * 左括号必须以正确的顺序闭合。
     */
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        Map<Character,Character> map = new HashMap<>();
        map.put('(',')');
        map.put('[',']');
        map.put('{','}');
        for (int i = 0; i < s.length(); i++) {
            if (stack.isEmpty() || s.charAt(i) != map.get(stack.peek())){
                if (!map.containsKey(s.charAt(i))){
                    return false;
                }
                stack.add(s.charAt(i));
            }else {
                stack.pop();
            }
        }
        return stack.isEmpty() ? true : false;
    }

    public boolean isValid2(String s) {
        Stack<Character> stack = new Stack<>();
        Map<Character, Character> map = new HashMap<>();
        map.put('(', ')');
        map.put('[', ']');
        map.put('{', '}');
        for (char c : s.toCharArray()) {
            if (map.containsKey(c)){
                stack.push(c);
            }else if (!stack.isEmpty() && map.get(stack.peek()) == c){
                stack.pop();
            }else {
                return false;
            }
        }
        return stack.isEmpty();
    }

    /**
     * 21. 合并两个有序链表
     * 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
     */
    public ListNode mergeTwoLists21(ListNode list1, ListNode list2) {
        ListNode head = new ListNode();
        ListNode tag = head;
        tag.next = null;
        ListNode p = list1;
        ListNode q = list2;
        while (p != null && q != null){
            if (p.val < q.val){
                list1 = list1.next;
                p.next = tag.next;
                tag.next = p;
                tag = p;
                p = list1;
            }else {
                list2 = list2.next;
                q.next = tag.next;
                tag.next = q;
                tag = q;
                q = list2;
            }
        }
        if (p != null){
            tag.next = p;
        }
        if (q != null){
            tag.next = q;
        }
        return head.next;
    }

    public ListNode mergeTwoLists21_2(ListNode list1, ListNode list2) {
        ListNode p = new ListNode();
        ListNode q;
        p.next = list1;
        list1 = p;
        q = list2;
        while (p.next != null && q != null){
            if (p.next.val <= q.val){
                p = p.next;
            }else {
                list2 = q.next;
                q.next = p.next;
                p.next = q;
                q = list2;
                p = p.next;
            }
        }
        if (p.next == null){
            p.next = q;
        }
        return list1.next;
    }

    /**
     * 22. 括号生成
     * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
     */
    List<String> ansParent;
    public List<String> generateParenthesis(int n) {
        ansParent = new ArrayList<>();
        String str = "";
        backtrack(0, 0, str, n);
        return ansParent;
    }
    private void backtrack(int left, int right, String str, int n){
        if (str.length() == n * 2){
            ansParent.add(str);
            return;
        }
        if (left < n){
            backtrack(left + 1, right, str + "(", n);
        }
        if (left > right){
            backtrack(left, right + 1, str + ")", n);
        }
    }

    public List<String> generateParenthesis2(int n) {
        List<String> res = new ArrayList<>();
        backtrack(res, "", n, 0, 0);
        return res;
    }

    private void backtrack(List<String> res, String str, int n, int left, int right) {
        if (str.length() == n * 2){
            res.add(str);
            return;
        }
        if (left < n){
            backtrack(res, str + "(", n, left + 1, right);
        }
        if (right < left){
            backtrack(res, str + ")", n, left, right + 1);
        }
    }

    /**
     * 23. 合并K个升序链表
     * 给你一个链表数组，每个链表都已经按升序排列。
     *
     * 请你将所有链表合并到一个升序链表中，返回合并后的链表。
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0){
            return new ListNode();
        }
        if (lists.length == 1){
            return lists[0];
        }
        PriorityQueue<ListNode> pq = new PriorityQueue<>(((o1, o2) -> o1.val - o2.val));
        for (ListNode head : lists) {
            if (head != null){
                pq.offer(head);
            }
        }
        ListNode head = new ListNode();
        head.next = null;
        ListNode p = head;
        while (!pq.isEmpty()){
            ListNode node = pq.poll();
            if (node.next != null){
                pq.offer(node.next);
            }
            node.next = p.next;
            p.next = node;
            p = node;
        }
        return head.next;
    }

    /**
     * 24. 两两交换链表中的节点
     * 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
     */
    public ListNode swapPairs(ListNode head) {
        ListNode p = new ListNode();
        p.next = head;
        ListNode q = head;
        head = p;
        while (q != null && q.next != null){
            p.next = q.next;
            p = q.next;
            q.next = p.next;
            p.next = q;
            p = q;
            q = q.next;
        }
        return head.next;
    }

    /**
     * 26. 删除有序数组中的重复项
     * 给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。
     *
     * 由于在某些语言中不能改变数组的长度，所以必须将结果放在数组nums的第一部分。更规范地说，如果在删除重复项之后有 k 个元素，那么nums的前 k 个元素应该保存最终结果。
     *
     * 将最终结果插入nums 的前 k 个位置后返回 k 。
     *
     * 不要使用额外的空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
     */
    public int removeDuplicates(int[] nums) {
        int count = 0;
        int temp = nums[0];
        for (int i = 0; i < nums.length; i++) {
            if (temp != nums[i]){
                count++;
                temp = nums[i];
            }
            nums[count] = nums[i];
        }
        return count + 1;
    }

    /**
     * 27. 移除元素
     * 给你一个数组 nums和一个值 val，你需要 原地 移除所有数值等于val的元素，并返回移除后数组的新长度。
     *
     * 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
     *
     * 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
     */
    public int removeElement(int[] nums, int val) {
        int count = -1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != val){
                count++;
                nums[count] = nums[i];
            }
        }
        return ++count;
    }

    /**
     * 28. 实现 strStr()
     * 实现strStr()函数。
     *
     * 给你两个字符串haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回 -1 。
     *
     * 说明：
     *
     * 当needle是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。
     *
     * 对于本题而言，当needle是空字符串时我们应当返回 0 。这与 C 语言的strstr()以及 Java 的indexOf()定义相符。
     */
    public int strStr(String haystack, String needle) {
        if (needle.isEmpty()){
            return 0;
        }
        int pre = 0;
        int rear = needle.length();
        while (rear <= haystack.length()){
            String substring = haystack.substring(pre, rear);
            if (haystack.substring(pre++,rear++).equals(needle)){
                return --pre;
            }
        }
        return -1;
    }

    public int strStr2(String haystack, String needle) {
        if (needle.length() == 0){
            return 0;
        }
        int[] next = new int[needle.length()];
        getNext(next, needle);
        int j = 0;
        for (int i = 0; i < haystack.length(); i++) {
            while (j > 0 && haystack.charAt(i) != needle.charAt(j)){
                j = next[j - 1];
            }
            if (haystack.charAt(i) == needle.charAt(j)){
                j++;
            }
            if (j == needle.length()){
                return i - needle.length() + 1;
            }
        }
        return -1;
    }

    private void getNext(int[] next, String s){
        int j = 0;
        next[0] = 0;
        for (int i = 1; i < s.length(); i++) {
            while (j > 0 && s.charAt(i) != s.charAt(j)){
                j = next[j - 1];
            }
            if (s.charAt(i) == s.charAt(j)){
                j++;
            }
            next[i] = j;
        }
    }

    /**
     * 31. 下一个排列
     * 整数数组的一个 排列 就是将其所有成员以序列或线性顺序排列。
     *
     * 例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。
     * 整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，
     * 那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，
     * 那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。
     *
     * 例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
     * 类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
     * 而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
     * 给你一个整数数组 nums ，找出 nums 的下一个排列。
     *
     * 必须 原地 修改，只允许使用额外常数空间。
     */
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]){
            i--;
        }
        if (i >= 0){
            int j = nums.length - 1;
            while (j >= 0 && nums[j] <= nums[i]){
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }
    private void reverse(int[] nums, int start){
        int left = start, right = nums.length - 1;
        while (left < right){
            swap(nums, left++, right--);
        }
    }



    /**
     * 33：搜索旋转排序数组
     * 整数数组 nums 按升序排列，数组中的值 互不相同 。
     * <p>
     * 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
     * <p>
     * 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
     * <p>
     *  
     * <p>
     * 示例 1：
     * <p>
     * 输入：nums = [4,5,6,7,0,1,2], target = 0
     * 输出：4
     * 示例 2：
     * <p>
     * 输入：nums = [4,5,6,7,0,1,2], target = 3
     * 输出：-1
     * <p>
     * [6,7,0,1,2,3,4,5]
     * <p>
     * 示例 3：
     * <p>
     * 输入：nums = [1], target = 0
     * 输出：-1
     */
    public int searchVolvo(int[] nums, int target) {
        int low = 0;
        int high = nums.length - 1;
        int mid;
        while (low <= high) {
            mid = (low + high) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[low] <= nums[mid]) {
                if (nums[low] <= target && nums[mid] > target) {
                    high = mid - 1;
                } else {
                    low = mid + 1;
                }
            } else {
                if (nums[mid] < target && nums[high] >= target) {
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
        }
        return -1;
    }

    public int searchVolvo33(int[] nums, int target) {
        if (nums.length == 0){
            return -1;
        }
        if (nums.length == 1){
            return nums[0] == target ? 0 : -1;
        }
        int low = 0;
        int high = nums.length - 1;
        while (low <= high){
            int mid = low + (high - low) / 2;
            if (nums[mid] == target){
                return mid;
            }
            if (nums[0] <= nums[mid]){
                if (nums[0] <= target && target < nums[mid]){
                    high = mid - 1;
                }else {
                    low = mid + 1;
                }
            }else {
                if (nums[mid] < target && target <= nums[high]){
                    low = mid + 1;
                }else {
                    high = mid - 1;
                }
            }
        }
        return -1;
    }

    /**
     * 34 ： 在排序数组中查找元素的第一个和最后一个位置
     * 给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
     * <p>
     * 如果数组中不存在目标值 target，返回 [-1, -1]。
     */
    public int[] searchRange(int[] nums, int target) {
        int[] result = {-1, -1};
        if (nums.length == 0) {
            return result;
        }
        int low = 0;
        int high = nums.length - 1;
        int mid;

        while (low <= high) {
            mid = (low + high) / 2;
            if (nums[mid] == target) {
                int start = mid;
                int end = mid;
                int i = 1;
                while (mid + i < nums.length && nums[mid + i] == target || mid - i > -1 && nums[mid - i] == target) {
                    if (mid + i < nums.length && nums[mid + i] == target) {
                        end = mid + i;
                    }
                    if (mid - i > -1 && nums[mid - i] == target) {
                        start = mid - i;
                    }
                    i++;
                }
                int[] m = {start, end};
                return m;
            } else if (nums[mid] > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return result;
    }

    public int[] searchRange34(int[] nums, int target) {
        int leftIndex = binarySearch(nums, target, true);
        int rightIndex = binarySearch(nums, target, false) - 1;
        if (leftIndex <= rightIndex && rightIndex < nums.length && nums[leftIndex] == target && nums[rightIndex] == target){
            return new int[]{leftIndex, rightIndex};
        }
        return new int[]{-1, -1};
    }

    private int binarySearch(int[] nums, int target, boolean left) {
        int low = 0;
        int high = nums.length - 1;
        int res = nums.length;
        while (low <= high){
            int mid = low + (high - low) / 2;
            if (nums[mid] > target || left && nums[mid] >= target){
                high = mid - 1;
                res = mid;
            }else {
                low = mid + 1;
            }
        }
        return res;
    }


    /**
     * 35. 搜索插入位置
     * 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
     *
     * 请必须使用时间复杂度为 O(log n) 的算法。
     */
    public int searchInsert(int[] nums, int target) {
        if (target < nums[0]){
            return 0;
        }else if (target > nums[nums.length - 1]){
            return nums.length;
        }
        int low = 0;
        int high = nums.length - 1;
        int mid = 0;
        while (low <= high){
            mid = (low + high) / 2;
            if (nums[mid] == target){
                return mid;
            }else if(nums[mid] < target){
                low = mid + 1;
            }else {
                high = mid - 1;
            }
        }
        return low;
    }

    /**
     * 37. 解数独
     * 编写一个程序，通过填充空格来解决数独问题。
     *
     * 数独的解法需 遵循如下规则：
     *
     * 数字1-9在每一行只能出现一次。
     * 数字1-9在每一列只能出现一次。
     * 数字1-9在每一个以粗实线分隔的3x3宫内只能出现一次。（请参考示例图）
     * 数独部分空格内已填入了数字，空白格用'.'表示。
     */
    public void solveSudoku(char[][] board) {
        helper(board);
    }

    private boolean helper(char[][] board){
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.'){
                    for (char num = '1'; num <= '9'; num++){
                        if (isValid(board, i, j, num)){
                            board[i][j] = num;
                            if (helper(board)){
                                return true;
                            }
                            board[i][j] = '.';
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

    private boolean isValid(char[][] board, int row, int col, char num) {
        for (int i = 0; i < 9; i++) {
            if (board[i][col] == num){
                return false;
            }
            if (board[row][i] == num){
                return false;
            }
            if (board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == num){
                return false;
            }
        }
        return true;
    }



    /**
     * 39. 组合总和
     * 给你一个 无重复元素 的整数数组candidates 和一个目标整数target，
     * 找出candidates中可以使数字和为目标数target 的 所有不同组合 ，
     * 并以列表形式返回。你可以按 任意顺序 返回这些组合。
     *
     * candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。
     *
     * 对于给定的输入，保证和为target 的不同组合数少于 150 个。
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack(candidates, target, res, 0, new ArrayList<Integer>());
        return res;
    }

    private void backtrack(int[] candidates, int target, List<List<Integer>> res, int i, ArrayList<Integer> temp_list) {
        if (target < 0){
            return;
        }else if (target == 0){
            res.add(new ArrayList<>(temp_list));
            return;
        }
        for (int start = i; start < candidates.length; start++) {
            if (target < 0){
                break;
            }
            temp_list.add(candidates[start]);
            backtrack(candidates, target - candidates[start], res, start, temp_list);
            temp_list.remove(temp_list.size() - 1);
        }
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack(res, candidates, new ArrayList<Integer>(), target, 0);
        return res;
    }

    private void backtrack(List<List<Integer>> res, int[] candidates, ArrayList<Integer> curPath, int target, int index) {
        if (target == 0){
            res.add(new ArrayList<>(curPath));
            return;
        }
        for (int i = index; i < candidates.length; i++) {
            if (candidates[i] > target){
                break;
            }
            curPath.add(candidates[i]);
            backtrack(res, candidates, curPath, target - candidates[i], i);
            curPath.remove(curPath.size() - 1);
        }
    }

    //法二
    public List<List<Integer>> combinationSum_Ⅱ(int[] candidates, int target) {
        int len = candidates.length;
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0){
            return res;
        }
        //排序是剪枝的前提
        Arrays.sort(candidates);
        Deque<Integer> path = new ArrayDeque<>();
        dfs(candidates, 0, len, target, path, res);
        return res;
    }

    /**
     *  深度优先遍历
     * @param candidates    候选数组
     * @param begin         搜索起点
     * @param len           冗余变量，是 candidates 里的属性，可以不传
     * @param target        每减去一个元素，目标值变小
     * @param path          从根结点到叶子结点的路径，是一个栈
     * @param res           结果集列表
     */
    private void dfs(int[] candidates, int begin, int len, int target, Deque<Integer> path, List<List<Integer>> res) {
        // 由于进入更深层的时候，小于 0 的部分被剪枝，因此递归终止条件值只判断等于 0 的情况
        if (target == 0){
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = begin; i < len; i++) {
            //剪枝，前提是数组有序
            if (target - candidates[i] < 0){
                break;
            }
            path.addLast(candidates[i]);
            dfs(candidates, i, len, target - candidates[i], path, res);
            path.removeLast();
        }
    }

    public List<List<Integer>> combinationSum3(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates == null || candidates.length == 0){
            return res;
        }
        Arrays.sort(candidates);
        reverse39(candidates);
        backtrack(candidates, res, new LinkedList<Integer>(), 0, target);
        return res;
    }

    private void backtrack(int[] candidates, List<List<Integer>> res, LinkedList<Integer> curPath, int index, int target) {
        if (target == 0){
            res.add(new ArrayList<>(curPath));
            return;
        }
        if (target > 0 && index < candidates.length){
            backtrack(candidates, res, curPath, index + 1, target);
            curPath.add(candidates[index]);
            backtrack(candidates, res, curPath, index, target - candidates[index]);
            curPath.removeLast();
        }

    }

    private void reverse39(int[] nums){
        int low = 0;
        int high = nums.length - 1;
        while (low < high){
            int temp = nums[low];
            nums[low++] = nums[high];
            nums[high--] = temp;
        }
    }

    /**
     * 40. 组合总和 II
     * 给定一个候选人编号的集合candidates和一个目标数target，找出candidates中所有可以使数字和为target的组合。
     *
     * candidates中的每个数字在每个组合中只能使用一次。
     *
     * 注意：解集不能包含重复的组合。
     */
    public List<List<Integer>> combinationSum4(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack(res, candidates, target, 0, new ArrayList<Integer>());
        return res;
    }

    private void backtrack(List<List<Integer>> res, int[] candidates, int target, int index, ArrayList<Integer> curPath) {
        if (target == 0){
            res.add(new ArrayList<>(curPath));
            return;
        }
        for (int i = index; i < candidates.length; i++) {
            if (i > index && candidates[i] == candidates[i - 1] || target - candidates[i] < 0){
                continue;
            }
            curPath.add(candidates[i]);
            backtrack(res, candidates, target - candidates[i], i + 1, curPath);
            curPath.remove(curPath.size() - 1);
        }
    }

    /**
     * 45. 跳跃游戏 II
     * 给你一个非负整数数组nums ，你最初位于数组的第一个位置。
     *
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     *
     * 你的目标是使用最少的跳跃次数到达数组的最后一个位置。
     *
     * 假设你总是可以到达数组的最后一个位置。
     */
    public int jump(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, nums.length);
        dp[0] = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j <= Math.min(i + nums[i], nums.length - 1); j++) {
                dp[j] = Math.min(dp[j], dp[i] + 1);
                if (j == nums.length - 1){
                    break;
                }
            }
        }
        return dp[nums.length - 1];
    }

    public int jump2(int[] nums) {
        int end = 0, maxPositions = 0, steps = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            maxPositions = Math.max(maxPositions, i + nums[i]);
            if (i == end){
                end = maxPositions;
                steps++;
            }
        }
        return steps;
    }

    /**
     * 46. 全排列
     * 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
     */
    List<List<Integer>> resPermute;
    public List<List<Integer>> permute(int[] nums) {
        if (nums.length == 0){
            return new ArrayList<>();
        }
        resPermute = new ArrayList<>();
        int[] visited = new int[nums.length];
        backtrack(nums, new ArrayList<Integer>(), visited);
        return resPermute;
    }

    private void backtrack(int[] nums, ArrayList<Integer> temp, int[] visited) {
        if (temp.size() == nums.length){
            resPermute.add(new ArrayList<>(temp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i] == 1){
                continue;
            }
            visited[i] = 1;
            temp.add(nums[i]);
            backtrack(nums, temp, visited);
            visited[i] = 0;
            temp.remove(temp.size() - 1);
        }
    }

    /**
     * 47. 全排列 II
     * @param nums      可包含重复数字的序列 num
     * @return          所有不重复的全排列
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        boolean[] visited = new boolean[nums.length];
        backtrack(res, nums, visited, new ArrayList<Integer>(), 0);
        return res;
    }

    private void backtrack(List<List<Integer>> res, int[] nums, boolean[] visited, ArrayList<Integer> curPath, int index) {
        if (curPath.size() == nums.length){
            res.add(new ArrayList<>(curPath));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i] || i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]){
                continue;
            }
            curPath.add(nums[i]);
            visited[i] = true;
            backtrack(res, nums, visited, curPath, i + 1);
            curPath.remove(curPath.size() - 1);
            visited[i] = false;
        }
    }

    /**
     * 48. 旋转图像
     * 给定一个 n×n 的二维矩阵matrix 表示一个图像。请你将图像顺时针旋转 90 度。
     *
     * 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
     */
    public void rotate(int[][] matrix) {
        //水平翻转
        for (int i = 0; i < matrix.length / 2; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[matrix.length - i - 1][j];
                matrix[matrix.length - i - 1][j] = temp;
            }
        }
        //沿主对角线翻转
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }

    /**
     * 52. N皇后 II
     * n皇后问题 研究的是如何将 n个皇后放置在 n × n 的棋盘上，并且使皇后彼此之间不能相互攻击。
     *
     * 给你一个整数 n ，返回 n 皇后问题 不同的解决方案的数量。
     */
    int countNQueens;
    public int totalNQueens(int n) {
        countNQueens = 0;
        backtrack(n, 0, new int[n]);
        return countNQueens;
    }

    private void backtrack(int n, int row, int[] columns) {
        //是否在所有行里都拜访了Queen？
        if (row == n){
            countNQueens++;
            return;
        }
        //尝试着将Queen放置在当前行中的每一列
        for (int col = 0; col < n; col++) {
            columns[row] = col;
            if (check(row, col, columns)){
                backtrack(n, row + 1, columns);
            }
            //如果不合法，回溯
            columns[row] = -1;
        }
    }

    private boolean check(int row, int col, int[] columns){
        for (int r = 0; r < row; r++) {
            if (columns[r] == col || row - r == Math.abs(columns[r] - col)){
                return false;
            }
        }
        return true;
    }

    /**
     * 55. 跳跃游戏
     * 给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
     *
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     *
     * 判断你是否能够到达最后一个下标。
     */
    public boolean canJump(int[] nums) {
        int rightMost = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i <= rightMost){
                rightMost = Math.max(rightMost, i + nums[i]);
                if (rightMost >= nums.length - 1){
                    return true;
                }
            }
        }
        return false;
    }

    public boolean canJump2(int[] nums) {
        int rightMost = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i <= rightMost){
                rightMost = Math.max(rightMost, i + nums[i]);
                if (rightMost >= nums.length - 1){
                    return true;
                }
            }else {
                break;
            }
        }
        return false;
    }

    /**
     * 56. 合并区间
     * 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
     * 请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。
     */
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0){
            return new int[0][2];
        }
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });
        List<int[]> merged = new ArrayList<>();
        for (int i = 0; i < intervals.length; i++) {
            int L = intervals[i][0], R = intervals[i][1];
            if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < L){
                merged.add(new int[]{L, R});
            } else {
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], R);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }

    /**
     * 58. 最后一个单词的长度
     * 给你一个字符串 s，由若干单词组成，单词前后用一些空格字符隔开。返回字符串中 最后一个 单词的长度。
     *
     * 单词 是指仅由字母组成、不包含任何空格字符的最大子字符串。
     */
    public int lengthOfLastWord(String s) {
        String[] str = s.split("\\s+");
        return str[str.length - 1].length();
    }

    /**
     * 59. 螺旋矩阵 II
     * 给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。
     */
    public int[][] generateMatrix(int n) {
        int[][] matrix = new int[n][n];
        int[] directions = {0, 1, 0, -1, 0};
        int[] dirs = {1, -1, -1, 1};
        int row = 0, col = 0, temp = 1;
        int len = (int) Math.pow(n, 2);
        int i = 0, t = 0;
        while (temp <= len) {
            while (row >= 0 && col >= 0 && row < n && col < n && matrix[row][col] == 0){
                matrix[row][col] = temp++;
                row += directions[t];
                col += directions[t + 1];
            }
            row += dirs[(i + 4) % 4];
            col += dirs[(i + 1 + 4) % 4];
            i++;
            t = (t + 1) % 4;
        }
        return matrix;
    }

    /**
     * 62. 不同路径
     * 一个机器人位于一个 m x n网格的左上角 （起始点在下图中标记为 “Start” ）。
     *
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
     *
     * 问总共有多少条不同的路径？
     */
    public int uniquePaths(int m, int n) {
        int[][] grid = new int[m][n];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (i == 0 || j == 0){
                    grid[i][j] = 1;
                }else {
                    grid[i][j] = grid[i - 1][j] + grid[i][j - 1];
                }
            }
        }
        return grid[m - 1][n - 1];
    }

    public int uniquePaths2(int m, int n) {
        int[][] matrix = new int[m][n];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (i == 0 || j == 0){
                    matrix[i][j] = 1;
                }else {
                    matrix[i][j] = matrix[i - 1][j] + matrix[i][j - 1];
                }
            }
        }
        return matrix[m - 1][n - 1];
    }

    /**
     * 63. 不同路径 II
     * 一个机器人位于一个m x n网格的左上角 （起始点在下图中标记为 “Start” ）。
     *
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish”）。
     *
     * 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
     *
     * 网格中的障碍物和空位置分别用 1 和 0 来表示。
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        for (int row = 0; row < obstacleGrid.length; row++) {
            for (int col = 0; col < obstacleGrid[0].length; col++) {
                if (obstacleGrid[row][col] == 1){
                    obstacleGrid[row][col] = -1;
                    continue;
                }
                if (row == 0 && col == 0){
                    obstacleGrid[row][col] = 1;
                } else if (row == 0){
                    obstacleGrid[row][col] = obstacleGrid[row][col - 1];
                }else if (col == 0){
                    obstacleGrid[row][col] = obstacleGrid[row - 1][col];
                }else {
                    obstacleGrid[row][col] += Math.max(obstacleGrid[row - 1][col], 0);
                    obstacleGrid[row][col] += Math.max(obstacleGrid[row][col - 1], 0);
                }
            }
        }
        return Math.max(obstacleGrid[obstacleGrid.length - 1][obstacleGrid[0].length - 1], 0);
    }



    /**
     * 64. 最小路径和
     * 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     *
     * 说明：每次只能向下或者向右移动一步。
     */
    public int minPathSum(int[][] grid) {
        for (int row = 0; row < grid.length; row++) {
            for (int col = 0; col < grid[0].length; col++) {
                if (row == 0 && col == 0){
                    continue;
                }else if (row == 0){
                    grid[row][col] += grid[row][col - 1];
                }else if (col == 0){
                    grid[row][col] += grid[row - 1][col];
                }else {
                    grid[row][col] += Math.min(grid[row - 1][col],  grid[row][col - 1]);
                }
            }
        }
        return grid[grid.length - 1][grid[0].length - 1];
    }

    /**
     * 66. 加一
     * 给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。
     *
     * 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
     *
     * 你可以假设除了整数 0 之外，这个整数不会以零开头。
     */
    public int[] plusOne(int[] digits) {
        int t = digits.length - 1;
        int p = 1;
        int temp;
        while (p == 1){
            temp = (digits[t] + 1) % 10;
            p = (digits[t] + 1) / 10;
            digits[t--] = temp;
            if (t == -1 && p == 1){
                int[] ans = new int[digits.length + 1];
                ans[0] = 1;
                for (int i = 0; i < digits.length; i++) {
                    ans[i + 1] = digits[i];
                }
                return ans;
            }
        }
        return digits;

    }

    /**
     * 67. 二进制求和
     * 给你两个二进制字符串，返回它们的和（用二进制表示）。
     *
     * 输入为 非空 字符串且只包含数字 1 和 0。
     */
    public String addBinary(String a, String b) {
        StringBuffer sb = new StringBuffer();
        int p = 0;
        int i = 0;
        while (i < a.length() || i < b.length()){
            int add = (i < a.length() ? a.charAt(a.length() - i - 1) - '0' : 0) + (i < b.length() ? b.charAt(b.length() - i - 1) - '0' : 0) + p;
            sb.append(add % 2);
            p = add / 2;
            i++;
        }
        if (p == 1){
            sb.append(1);
        }
        return sb.reverse().toString();
    }

    /**
     * 69. x 的平方根
     * 给你一个非负整数 x ，计算并返回x的 算术平方根 。
     *
     * 由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。
     *
     * 注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 。
     */
    public int mySqrt(int x) {
        int i = 0;
        int j = x;
        int ans = -1;
        while (i <= j){
            int mid = (i + j) / 2;
            if (mid * mid <= x){
                ans = mid;
                i = mid + 1;
            }else {
                j = mid - 1;
            }
        }
        return ans;
    }
    public int mySqrt1(int x){
        if (x == 0) {
            return 0;
        }

        double C = x, x0 = x;
        while (true) {
            double xi = 0.5 * (x0 + C / x0);
            if (Math.abs(x0 - xi) < 1e-7) {
                break;
            }
            x0 = xi;
        }
        return (int) x0;
    }

    /**
     * 70. 爬楼梯
     * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
     *
     * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     */
    //非递归
    public int climbStairs(int n) {
        if (n == 0 || n == 1){
            return n;
        }
        int p = 1;
        int q = 1;
        int temp;
        for (int i = 2; i <= n; i++) {
            temp = p + q;
            p = q;
            q = temp;
        }
        return q;
    }

    //递归
    public int climbStairs_recursion(int n) {
        int[] cache = new int[n + 1];
        return steps(n, cache);
    }

    private int steps(int n, int[] cache) {
        if (n == 1){
            cache[1] = 1;
            return 1;
        }
        if (n == 2){
            cache[2] = 2;
            return 2;
        }
        if (cache[n] != 0){
            return cache[n];
        }
        cache[n] = steps(n - 1, cache) + steps(n - 2, cache);
        return cache[n];
    }

    /**
     * 72. 编辑距离
     * 给你两个单词word1 和word2， 请返回将word1转换成word2 所使用的最少操作数 。
     *
     * 你可以对一个单词进行如下三种操作：
     *
     * 插入一个字符
     * 删除一个字符
     * 替换一个字符
     */
    public int minDistance72(String word1, String word2) {
        int M = word1.length();
        int N = word2.length();
        int[][] dp = new int[M + 1][N + 1];
        for (int i = 1; i <= M; i++) {
            dp[i][0] = i;
        }
        for (int i = 1; i <= N; i++) {
            dp[0][i] = i;
        }
        for (int i = 1; i <= M; i++) {
            char c1 = word1.charAt(i - 1);
            for (int j = 1; j <= N; j++) {
                if (c1 == word2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1];
                }else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                }
            }
        }
        return dp[M][N];
    }

    /**
     * 74: 搜索二维矩阵
     * 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
     * <p>
     * 每行中的整数从左到右按升序排列。
     * 每行的第一个整数大于前一行的最后一个整数。
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0) {
            return false;
        }
        if (matrix[0][0] == target) {
            return true;
        }
        int i = 0;
        int j = 0;
        while (i < matrix.length && j < matrix[0].length) {
            if (target == matrix[i][j]) {
                return true;
            } else if (i + 1 < matrix.length && target >= matrix[i + 1][j]) {
                i++;
            } else if (j + 1 < matrix[0].length && target >= matrix[i][j + 1]) {
                j++;
            } else {
                return false;
            }
        }
        return false;
    }

    //两次二分查找
    public boolean searchMatrix1(int[][] matrix, int target) {
        int rowIndex = binarySearchFirstColumn(matrix, target);
        if (rowIndex < 0) {
            return false;
        }
        return binarySearchRow(matrix[rowIndex], target);
    }

    public int binarySearchFirstColumn(int[][] matrix, int target) {
        int low = -1, high = matrix.length - 1;
        while (low < high) {
            int mid = (high - low + 1) / 2 + low;
            if (matrix[mid][0] <= target) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }

    public boolean binarySearchRow(int[] row, int target) {
        int low = 0, high = row.length - 1;
        while (low <= high) {
            int mid = (high - low) / 2 + low;
            if (row[mid] == target) {
                return true;
            } else if (row[mid] > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return false;
    }

    public boolean searchMatrix74(int[][] matrix, int target) {
        if (matrix.length == 0){
            return false;
        }else if (matrix.length == 1){
            if (matrix[0].length == 0){
                return false;
            }else if (matrix[0].length == 1 && matrix[0][0] == target){
                return true;
            }
        }
        int row = 0;
        row = binarySearch(matrix, target, true, row);
        if (row == -1){
            return false;
        }
        int col = 0;
        col = binarySearch(matrix, target, false, row);
        return col > -1 ? true : false;
    }

    private int binarySearch(int[][] matrix, int target, boolean isRow, int rowIndex){
        int low = 0;
        int high;
        if (isRow){
            high = matrix.length - 1;
            while (low <= high){
                int mid = low + (high - low) / 2;
                if (matrix[mid][0] == target){
                    return mid;
                }else if (matrix[mid][0] > target){
                    high = mid - 1;
                }else {
                    low = mid + 1;
                }
            }
            return high;
        }else {
            high = matrix[rowIndex].length - 1;
            while (low <= high){
                int mid = low + (high - low) / 2;
                if (matrix[rowIndex][mid] == target){
                    return mid;
                }else if (matrix[rowIndex][mid] > target){
                    high = mid - 1;
                }else {
                    low = mid + 1;
                }
            }
        }
        return -1;
    }



    /**
     * 75. 颜色分类
     * 给定一个包含红色、白色和蓝色、共n 个元素的数组nums，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
     *
     * 我们使用整数 0、1 和 2 分别表示红色、白色和蓝色。
     *
     * 必须在不使用库的sort函数的情况下解决这个问题。
     */
    public void sortColors(int[] nums) {
        if (nums.length < 2){
            return;
        }
        int low = 0;
        int zero = -1;
        int high = nums.length - 1;
        while (low <= high){
            if (nums[low] == 0){
                zero++;
                swap(nums, low++, zero);
            }else if (nums[low] == 1){
                low++;
            }else {
                swap(nums, low, high--);
            }
        }
    }


    /**
     * 77. 组合
     * 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
     *
     * 你可以按 任何顺序 返回答案。
     */
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        backtrack(res, new ArrayList<Integer>(), n, k, 1);
        return res;
    }

    private void backtrack(List<List<Integer>> res, ArrayList<Integer> path, int n, int k, int start) {
        if (path.size() == k){
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = start; i <= n; i++) {
            path.add(i);
            backtrack(res, path, n, k, i + 1);
            path.remove(path.size() - 1);
        }
    }

    /**
     * 78. 子集
     *给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
     *
     * 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
     */
    List<List<Integer>> resSubsets;
    public List<List<Integer>> subsets(int[] nums) {
        resSubsets = new ArrayList<>();
        backtrack(nums, new ArrayList<Integer>(), 0);
        return resSubsets;
    }

    private void backtrack(int[] nums, ArrayList<Integer> path, int start) {
        resSubsets.add(new ArrayList<>(path));
        for (int i = start; i < nums.length; i++) {
            path.add(nums[i]);
            backtrack(nums,path, i + 1);
            path.remove(path.size() - 1);
        }
    }

    public List<List<Integer>> subsets2(int[] nums) {
        resSubsets = new ArrayList<>();
        backtrack2(nums, new ArrayList<Integer>(), 0);
        return resSubsets;
    }

    private void backtrack2(int[] nums, ArrayList<Integer> curPath, int start) {
        resSubsets.add(new ArrayList<>(curPath));
        for (int i = start; i < nums.length; i++) {
            curPath.add(nums[i]);
            backtrack2(nums, curPath, i + 1);
            curPath.remove(curPath.size() - 1);
        }
    }

    /**
     * 79. 单词搜索
     * 给定一个m x n 二维字符网格board 和一个字符串单词word 。如果word 存在于网格中，返回 true ；否则，返回 false 。
     *
     * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
     */
    public boolean exist79(char[][] board, String word) {
        boolean[][] visited = new boolean[board.length][board[0].length];
        int[] directions = {0, 1, 0, -1, 0};
        for (int row = 0; row < board.length; row++) {
            for (int col = 0; col < board[0].length; col++) {
                if (backtrack(board, word, row, col, visited, directions, 0)){
                    return true;
                }
            }
        }
        return false;
    }

    private boolean backtrack(char[][] board, String word, int row, int col, boolean[][] visited, int[] directions, int len) {
        if (len == word.length()){
            return true;
        }
        if (row < 0 || row >= board.length || col < 0 || col >= board[0].length || visited[row][col] || board[row][col] != word.charAt(len)){
            return false;
        }
        visited[row][col] = true;
        for (int i = 0; i < directions.length - 1; i++) {
            int r = directions[i];
            int c = directions[i + 1];
            if (backtrack(board, word, row + r, col + c, visited, directions, len + 1)){
                return true;
            }
        }
        visited[row][col] = false;
        return false;
    }


    /**
     * 82-Ⅱ: 删除链表中的重复元素
     * 给定一个已排序的链表的头 head ， 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回 已排序的链表 。
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode headNode = new ListNode();
        ListNode p = head;
        ListNode q = headNode;

        int temp = 101;
        while (p.next != null) {
            head = p.next;
            if (p.val != temp && p.val != p.next.val) {
                q.next = p;
                p.next = null;
                q = q.next;
                p = head;
            } else {
                temp = p.val;
                p = p.next;
            }
        }
        if (p.val != temp) {
            q.next = p;
            p.next = null;
            q = q.next;
            p = head;
        }
        return headNode.next;
    }

    public ListNode deleteDuplicates2(ListNode head) {
        ListNode p = new ListNode();
        p.next = head;
        head = p;
        ListNode q;
        boolean flag = false;
        while (p.next != null){
            q = p.next;
            while (q.next != null && q.val == q.next.val){
                flag = true;
                q = q.next;
            }
            if (flag){
                p.next = q.next;
                flag = false;
            }else {
                p = p.next;
            }
        }
        return head.next;
    }

    /**
     * 83. 删除排序链表中的重复元素
     * 给定一个已排序的链表的头 head ， 删除所有重复的元素，使每个元素只出现一次 。返回 已排序的链表 。
     */
    public ListNode deleteDuplicates83(ListNode head) {
        if (head == null){
            return head;
        }
        ListNode p = head;
        while (p.next != null){
            if (p.val == p.next.val){
                p.next = p.next.next;
            }else {
                p = p.next;
            }
        }
        return head;
    }

    /**
     * 88. 合并两个有序数组
     * 给你两个按 非递减顺序 排列的整数数组nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。
     *
     * 请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。
     *
     * 注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。
     */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int k = nums1.length - 1;
        while (i >= 0 && j >= 0){
            nums1[k--] = nums1[i] >= nums2[j] ? nums1[i--] : nums2[j--];
        }
        while (j >= 0){
            nums1[k--] = nums2[j--];
        }
    }

    /**
     * 90. 子集 II
     * 给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。
     *
     * 解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        backtrack(res, nums, new ArrayList<Integer>(), 0);
        return res;
    }

    private void backtrack(List<List<Integer>> res, int[] nums, ArrayList<Integer> path, int start) {
        res.add(new ArrayList<>(path));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]){
                continue;
            }
            path.add(nums[i]);
            backtrack(res, nums, path, i + 1);
            path.remove(path.size() - 1);
        }
    }

    public List<List<Integer>> subsetsWithDup2(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        backtrack2(res, nums, new ArrayList<Integer>(), 0);
        return res;
    }

    private void backtrack2(List<List<Integer>> res, int[] nums, ArrayList<Integer> cur, int start) {
        res.add(new ArrayList<>(cur));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]){
                continue;
            }
            cur.add(nums[i]);
            backtrack2(res, nums, cur, i + 1);
            cur.remove(cur.size() - 1);
        }
    }

    /**
     * 91. 解码方法
     * 一条包含字母A-Z 的消息通过以下映射进行了 编码 ：
     *
     * 'A' -> "1"
     * 'B' -> "2"
     * ...
     * 'Z' -> "26"
     * 要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：
     *
     * "AAJF" ，将消息分组为 (1 1 10 6)
     * "KJF" ，将消息分组为 (11 10 6)
     * 注意，消息不能分组为 (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。
     *
     * 给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。
     *
     * 题目数据保证答案肯定是一个 32 位 的整数。
     */
    //递归实际无法使用，超时
    public int numDecodings(String s) {
        char[] chars = s.toCharArray();
        if(chars[0] == '0'){
            return 0;
        }
        return decode(chars, chars.length - 1);
    }

    private int decode(char[] chars, int index) {
        if (index <= 0){
            return 1;
        }
        int count = 0;
        char current = chars[index];
        int pre = chars[index - 1];
        if (current > '0'){
            count = decode(chars, index - 1);
        }
        if (pre == '1' || pre == '2' && current <= '6'){
            count += decode(chars, index - 2);
        }
        return count;
    }

    public int numDecodings_2(String s) {
        char[] chars = s.toCharArray();
        int n = chars.length;
        //a = f[i-2], b = f[i-1], c = f[i]
        int a = 0, b = 1, c= 0;
        for (int i = 1; i <= n; i++) {
            c = 0;
            if (chars[i - 1] != '0'){
                c += b;
            }
            if (i > 1 && chars[i - 2] != '0' && ((chars[i - 2] - '0') * 10 + (chars[i - 1] - '0')) <= 26){
                c += a;
            }
            a = b;
            b = c;
        }
        return c;
    }

    public int numDecodings3(String s) {
        if (s.length() <= 1){
            return 1;
        }
        char[] chars = s.toCharArray();
        //a = f[i-2], b = f[i-1], c = f[i]
        int a = 0, b = 1, c = 0;
        for (int i = 1; i <= chars.length; i++) {
            if (chars[i] > '2'){
                c = 0;
                if (chars[i - 1] != '0'){
                    c += b;
                }
                if (i > 1 && chars[i - 2] != '0' && ((chars[i - 2] - '0') * 10 + (chars[i - 1] - '0')) <= 26){
                    c += a;
                }
                a = b;
                b = c;
            }
        }
        return c;
    }

    public int numDecodings4(String s) {
        char[] chars = s.toCharArray();
        int n = chars.length;
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            if (chars[i - 1] != '0'){
                dp[i] += dp[i - 1];
            }
            if (i > 1 && chars[i - 2] != '0' && ((chars[i - 2] - '0') * 10 + (chars[i - 1] - '0')) <= 26){
                dp[i] += dp[i - 2];
            }
        }
        return dp[n];
    }

    public int numDecodings5(String s) {
        char[] chars = s.toCharArray();
        int a = 0, b = 1, c = 0;
        for (int i = 1; i < chars.length; i++) {
            c = 0;
            if (chars[i - 1] != '0'){
                c += b;
            }
            if (i > 1 && chars[i - 2] != '0' && ((chars[i - 2] - '0') * 10 + chars[i - 1] - '0') <= 26){
                c += a;
            }
            a = b;
            b = c;
        }
        return c;
    }



    /**
     * 94. 二叉树的中序遍历
     * 给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。
     */
    //递归
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        inOrder(root,ans);
        return ans;
    }
    private void inOrder(TreeNode root, List<Integer> ans){
        if (root == null){
            return;
        }
        inOrder(root.left,ans);
        ans.add(root.val);
        inOrder(root.right,ans);
    }
    //非递归
    public List<Integer> inorderTraversal2(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        Stack<TreeNode> queue = new Stack<>();
        TreeNode p = root;
        while (!queue.isEmpty() || p != null){
            if (p != null){
                queue.push(p);
                p = p.left;
            }else {
                ans.add(queue.peek().val);
                p = queue.pop().right;
            }
        }
        return ans;
    }

    /**
     * 96. 不同的二叉搜索树
     * 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。
     */
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        //当序列长度为 11（只有根）或为 00（空树）时，只有一种情况
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            //遍历数组，每个数作为一次根节点
            //G(n) = ΣF(i,n)                         (1)
            //当n为F(i,n) = G(i - 1) * G(n - i)       (2)
            //结合(1) (2) 得G(n) = ΣG(i-1) * G(n-i)
            for (int j = 1; j <= i; j++) {
                //给定序列 1 \cdots n1⋯n，我们选择数字 ii 作为根，
                // 则根为 ii 的所有二叉搜索树的集合是左子树集合和右子树集合的笛卡尔积，
                // 对于笛卡尔积中的每个元素，加上根节点之后形成完整的二叉搜索树
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

    //数学： 卡塔兰数
    public int numTreesCatalan(int n){
        //这里使用long类型防止计算过程中溢出
        long C = 1;
        for (int i = 0; i < n; i++) {
            C *= 2 * (2 * i + 1) / (i + 2);
        }
        return (int) C;
    }

    /**
     * 98. 验证二叉搜索树
     * 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
     *
     * 有效 二叉搜索树定义如下：
     *
     * 节点的左子树只包含 小于 当前节点的数。
     * 节点的右子树只包含 大于 当前节点的数。
     * 所有左子树和右子树自身必须也是二叉搜索树。
     */
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MAX_VALUE, Long.MIN_VALUE);
    }

    private boolean isValidBST(TreeNode root, long max, long min) {
        if (root == null){
            return true;
        }
        if (root.val >= max || root.val <= min){
            return false;
        }
        return isValidBST(root.left, root.val, min) && isValidBST(root.right, max, root.val);
    }

    /**
     * 100. 相同的树
     * 给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。
     *
     * 如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null ^ q == null){
            return false;
        }else if (p == null && q == null){
            return true;
        }else if (p.val == q.val){
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        }else {
            return false;
        }
    }

    /**
     * 101. 对称二叉树
     * 给你一个二叉树的根节点 root ， 检查它是否轴对称。
     */
    public boolean isSymmetric101(TreeNode root) {
        return check(root, root);
    }
    private boolean check(TreeNode p, TreeNode q){
        if (p == null && q == null){
            return true;
        }else if (p == null || q == null){
            return false;
        }else {
            return p.val == q.val && check(p.left, q.right) && check(p.right, q.left);
        }
    }

    /**
     * 102. 二叉树的层序遍历
     * 给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans = new LinkedList<>();
        if (root == null){
            return ans;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            List<Integer> list = new LinkedList<>();
            int len = queue.size();
            for (int i = 0; i < len; i++) {
                TreeNode node = queue.poll();
                list.add(node.val);
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
            ans.add(list);
        }
        return ans;
    }

    public List<List<Integer>> levelOrderC(TreeNode root) {
        List<List<Integer>> res = new LinkedList<>();
        if (root == null){
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int len = queue.size();
            List<Integer> cur = new LinkedList<>();
            for (int i = 0; i < len; i++) {
                TreeNode node = queue.poll();
                cur.add(node.val);
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
            res.add(cur);
        }
        return res;
    }

    /**
     * 104. 二叉树的最大深度
     * 给定一个二叉树，找出其最大深度。
     *
     * 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
     *
     * 说明: 叶子节点是指没有子节点的节点。
     */
    public int maxDepth104(TreeNode root) {
        if (root == null){
            return 0;
        }
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int ans = 0;
        while (!q.isEmpty()){
            int size = q.size();
            while (size > 0){
                TreeNode node = q.poll();
                if (node.left != null){
                    q.offer(node.left);
                }
                if (node.right != null){
                    q.offer(node.right);
                }
                size--;
            }
            ans++;
        }
        return ans;
    }

    /**
     * 105. 从前序与中序遍历序列构造二叉树
     * 给定两个整数数组preorder 和 inorder，其中preorder 是二叉树的先序遍历， inorder是同一棵树的中序遍历，请构造二叉树并返回其根节点。
     */
    public TreeNode buildTree105(int[] preorder, int[] inorder) {
        int len = preorder.length;
        return buildTreeHelper(preorder, 0, len, inorder, 0, len);
    }

    private TreeNode buildTreeHelper(int[] preorder, int pStart, int pEnd, int[] inorder, int iStart, int iEnd) {
        if (pStart == pEnd){
            return null;
        }
        int pRootVal = preorder[pStart];
        TreeNode root = new TreeNode(pRootVal);
        int iRootIndex = 0;
        for (int i = iStart; i < iEnd; i++) {
            if (inorder[i] == pRootVal){
                iRootIndex = i;
                break;
            }
        }
        int leftNum = iRootIndex - iStart;
        root.left = buildTreeHelper(preorder, pStart + 1, pStart + leftNum + 1, inorder, iStart, iRootIndex);
        root.right = buildTreeHelper(preorder, pStart + leftNum + 1, pEnd, inorder, iRootIndex + 1, iEnd);
        return root;
    }

    /**
     * 106. 从中序与后序遍历序列构造二叉树
     * 给定两个整数数组 inorder 和 postorder ，其中 inorder 是二叉树的中序遍历， postorder 是同一棵树的后序遍历，请你构造并返回这颗二叉树。
     */
    public TreeNode buildTree106(int[] inorder, int[] postorder) {
        int len = inorder.length;
        return buildTreeIPHelper(inorder, 0, len, postorder, 0, len);
    }

    private TreeNode buildTreeIPHelper(int[] inorder, int iStart, int iEnd, int[] postorder, int pStart, int pEnd) {
        if (pStart == pEnd){
            return null;
        }
        int pRootVal = postorder[pEnd - 1];
        TreeNode root = new TreeNode(pRootVal);
        int iRootIndex = 0;
        for (int i = 0; i < iEnd; i++) {
            if (inorder[i] == pRootVal){
                iRootIndex = i;
                break;
            }
        }
        int leftLen = iRootIndex - iStart;
        root.left = buildTreeIPHelper(inorder, iStart, iRootIndex, postorder, pStart, pStart + leftLen);
        root.right = buildTreeIPHelper(inorder, iRootIndex + 1, iEnd, postorder, pStart + leftLen, pEnd - 1);
        return root;
    }

    /**
     * 107. 二叉树的层序遍历 II
     * 给你二叉树的根节点 root ，返回其节点值 自底向上的层序遍历 。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new LinkedList<>();
        if (root == null){
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            List<Integer> cur = new LinkedList<>();
            int len = queue.size();
            for (int i = 0; i < len; i++) {
                TreeNode node = queue.poll();
                cur.add(node.val);
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
            res.add(0, cur);
        }
        return res;
    }


    /**
     * 108. 将有序数组转换为二叉搜索树
     * 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
     *
     * 高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        return dfs(nums, 0, nums.length - 1);
    }
    private TreeNode dfs(int[] nums, int low, int high){
        if (low > high){
            return null;
        }
        int mid = (low + high) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = dfs(nums, low, mid - 1);
        node.right = dfs(nums, mid + 1, high);
        return node;
    }

    /**
     * 110. 平衡二叉树
     *给定一个二叉树，判断它是否是高度平衡的二叉树。
     *
     * 本题中，一棵高度平衡二叉树定义为：
     *
     * 一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。
     */
    public boolean isBalanced_110(TreeNode root) {
        if (root == null){
            return true;
        }else {
            return Math.abs(height_110(root.left) - height_110(root.right)) <= 1 && isBalanced_110(root.left) && isBalanced_110(root.right);
        }

    }

    private int height_110(TreeNode root) {
        if (root == null){
            return 0;
        }else {
            return Math.max(height_110(root.left), height_110(root.right)) + 1;
        }
    }

    /**
     * 111. 二叉树的最小深度
     * 给定一个二叉树，找出其最小深度。
     *
     * 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
     *
     * 说明：叶子节点是指没有子节点的节点。
     */
    public int minDepth(TreeNode root) {
        if (root == null){
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int deep = 1;
        while (!queue.isEmpty()){
            int len = queue.size();
            for (int i = 0; i < len; i++) {
                TreeNode node = queue.poll();

                if (node.left == null && node.right == null){
                    return deep;
                }
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
            deep++;
        }
        return deep;
    }

    /**
     * 112. 路径总和
     * 给你二叉树的根节点root 和一个表示目标和的整数targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和targetSum 。如果存在，返回 true ；否则，返回 false 。
     *
     * 叶子节点 是指没有子节点的节点。
     */
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null){
            return false;
        }
        if (root.left == null && root.right == null){
            return targetSum == root.val;
        }
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }

    /**
     * 116. 填充每个节点的下一个右侧节点指针
     * 给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
     *
     * struct Node {
     *   int val;
     *   Node *left;
     *   Node *right;
     *   Node *next;
     * }
     *
     * 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
     *
     * 初始状态下，所有next 指针都被设置为 NULL。
     */
    class Node2 {
        public int val;
        public Node2 left;
        public Node2 right;
        public Node2 next;

        public Node2() {}

        public Node2(int _val) {
            val = _val;
        }

        public Node2(int _val, Node2 _left, Node2 _right, Node2 _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    };
    public Node2 connect(Node2 root) {
        if (root == null){
            return root;
        }
        Queue<Node2> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int len = queue.size();
            for (int i = 0; i < len; i++) {
                Node2 node = queue.poll();
                if (i < len - 1){
                    node.next = queue.peek();
                }
                if (node.left != null){
                    queue.offer(node.left);
                    queue.offer(node.right);
                }
            }
        }
        return root;
    }

    /**
     * 117. 填充每个节点的下一个右侧节点指针 II
     * 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
     *
     * 初始状态下，所有next 指针都被设置为 NULL。
     *
     * 进阶：
     *
     * 你只能使用常量级额外空间。
     * 使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。
     */
    public Node2 connect2(Node2 root) {
        if (root == null){
            return root;
        }
        Queue<Node2> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int len = queue.size();
            for (int i = 0; i < len; i++) {
                Node2 p = queue.poll();
                if (i == len - 1){
                    p.next = null;
                }else {
                    p.next = queue.peek();
                }
                if (p.left != null){
                    queue.offer(p.left);
                }
                if (p.right != null){
                    queue.offer(p.right);
                }
            }
        }
        return root;
    }

    /**
     * 118. 杨辉三角
     * 给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。
     *
     * 在「杨辉三角」中，每个数是它左上方和右上方的数的和。
     */
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> ans = new ArrayList<>();
        if (numRows == 0){
            return ans;
        }
        for (int i = 1; i <= numRows; i++) {
            List<Integer> list = new ArrayList<>();
            if (i == 1){
                list.add(1);
                ans.add(list);
            }else if (i == 2){
                list.add(1);
                list.add(1);
                ans.add(list);
            }else {
                list.add(1);
                List<Integer> temp = ans.get(i - 2);
                for (int j = 0; j < i - 2; j++) {
                    list.add(temp.get(j) + temp.get(j + 1));
                }
                list.add(1);
                ans.add(list);
            }
        }
        return ans;
    }

    /**
     * 119. 杨辉三角 II
     * 给定一个非负索引 rowIndex，返回「杨辉三角」的第 rowIndex 行。
     *
     * 在「杨辉三角」中，每个数是它左上方和右上方的数的和。
     */
    public List<Integer> getRow(int rowIndex) {
        List<Integer> row = new ArrayList<Integer>();
        row.add(1);
        for (int i = 1; i <= rowIndex; ++i) {
            row.add((int) ((long) row.get(i - 1) * (rowIndex - i + 1) / i));
        }
        return row;
    }

    /**
     * 120. 三角形最小路径和
     * 给定一个三角形 triangle ，找出自顶向下的最小路径和。
     *
     * 每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
     * 也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        Integer[][] cache = new Integer[triangle.size()][triangle.size()];
        return dfs(triangle, 0, 0, cache);
    }

    private int dfs(List<List<Integer>> triangle, int i, int j, Integer[][] cache) {
        if (i == triangle.size()){
            return 0;
        }
        if (cache[i][j] != null){
            return cache[i][j];
        }
        return cache[i][j] = Math.min(dfs(triangle, i + 1, j, cache), dfs(triangle, i + 1, j + 1, cache)) + triangle.get(i).get(j);
    }

    /**
     * 121. 买卖股票的最佳时机
     * 给定一个数组 prices ，它的第i 个元素prices[i] 表示一支给定股票第 i 天的价格。
     *
     * 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
     *
     * 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
     */
    public int maxProfit121(int[] prices) {
        int min = 100000;
        int profit = 0;
        for (int price : prices) {
            min = Math.min(min, price);
            profit = Math.max(profit, price - min);
        }
        return profit;
    }

    /**
     * 122: 买卖股票的最佳时机
     * 给你一个整数数组 prices ，其中prices[i] 表示某支股票第 i 天的价格。
     * <p>
     * 在每一天，你可以决定是否购买和/或出售股票。你在任何时候最多只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。
     * <p>
     * 返回 你能获得的 最大 利润。
     */
    public int maxProfit2(int[] prices) {
        int n = prices.length;
        int dp0 = 0, dp1 = -prices[0];
        for (int i = 1; i < n; ++i) {
            int newDp0 = Math.max(dp0, dp1 + prices[i]);
            int newDp1 = Math.max(dp1, dp0 - prices[i]);
            dp0 = newDp0;
            dp1 = newDp1;
        }
        return dp0;
    }

    /**
     * 125. 验证回文串
     * 给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
     *
     * 说明：本题中，我们将空字符串定义为有效的回文串。
     */
    public boolean isPalindrome(String s) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < s.length(); i++) {
            char temp = s.charAt(i);
            if (temp >= 48 && temp <= 57 || temp >= 65 && temp <= 90 || temp >= 97 && temp <= 122){
                if (temp >= 65 && temp <= 90){
                    temp += 32;
                }
                sb.append(temp);
            }
        }
        int i = 0;
        int j = sb.length() - 1;
        while (i < j) {
            if (sb.charAt(i++) != sb.charAt(j--)){
                return false;
            }
        }
        return true;
    }

    /**
     * 130. 被围绕的区域
     * 给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
     */
    public void solve(char[][] board) {
        int[] directions = {1, 0, -1, 0, 1};
        int r = board.length;
        int c = board[0].length;
        int[][] visited = new int[r][c];

        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                if ((i == 0 || i == r - 1 || j == 0 || j == c - 1) && board[i][j] == 'O'){
                    dfs(board, i, j, visited, directions, true);
                }
            }
        }
        for (int i = 1; i < r - 1; i++) {
            for (int j = 1; j < c - 1; j++) {
                if (board[i][j] == 'O' && visited[i][j] == 0){
                    dfs(board, i, j, visited, directions, false);
                }
            }
        }
    }

    private void dfs(char[][] board, int row, int col, int[][] visited, int[] directions, boolean brim) {
        Queue<int[]> queue = new LinkedList<>();
        visited[row][col] = 1;
        queue.offer(new int[]{row, col});
        while (!queue.isEmpty()){
            int[] node = queue.poll();
            row = node[0];
            col = node[1];
            if (!brim){
                board[row][col] = 'X';
            }
            for (int i = 0; i < directions.length - 1; i++) {
                row += directions[i];
                col += directions[i + 1];
                if (row < 0 || row == board.length || col < 0 || col == board[0].length || visited[row][col] == 1 || board[row][col] == 'X'){
                }else {
                    queue.offer(new int[]{row, col});
                    visited[row][col] = 1;
                }
                row -= directions[i];
                col -= directions[i + 1];
            }
        }
    }


    /**
     * 136. 只出现一次的数字
     * 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
     *
     * 说明：
     *
     * 你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
     */
    public int singleNumber(int[] nums) {
        int ans = 0;
        for (int num : nums) {
            ans ^= num;
        }
        return ans;
    }

    public int singleNumber136(int[] nums) {
        int res = 0;
        for (int num : nums) {
            res ^= num;
        }
        return res;
    }

    /**
     * 139. 单词拆分
     * 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
     *
     * 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 0; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && set.contains(s.substring(j, i))){
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    /**
     * 141. 环形链表
     * 给你一个链表的头节点 head ，判断链表中是否有环。
     *
     * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递。仅仅是为了标识链表的实际情况。
     *
     * 如果链表中存在环，则返回 true 。 否则，返回 false 。
     */
    public boolean hasCycle(ListNode head) {
        Set<ListNode> set = new HashSet<>();
        ListNode p  = head;
        while (p != null){
            if (set.contains(p)){
                return true;
            }else {
                set.add(p);
                p = p.next;
            }
        }
        return false;
    }

    /**
     * 142. 环形链表 II
     * 给定一个链表的头节点 head，返回链表开始入环的第一个节点。如果链表无环，则返回null。
     *
     * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。
     * 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。
     * 注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
     *
     * 不允许修改 链表。
     */
    public ListNode detectCycle(ListNode head) {
        Set<ListNode> set = new HashSet<>();
        ListNode p = head;
        while (p != null){
            if (set.contains(p)){
                return p;
            }else {
                set.add(p);
                p = p.next;
            }
        }
        return null;
    }

    public ListNode detectCycle2(ListNode head) {
        if (head == null){
            return null;
        }
        ListNode slow = head, fast = head;
        while (fast != null){
            slow = slow.next;
            if (fast.next != null){
                fast = fast.next.next;
            }else {
                return null;
            }
            if (fast == slow){
                ListNode ptr = head;
                while (ptr != slow){
                    ptr = ptr.next;
                    slow = slow.next;
                }
                return ptr;
            }
        }
        return null;
    }

    /**
     * 144. 二叉树的前序遍历
     * 给你二叉树的根节点 root ，返回它节点值的 前序 遍历。
     */
    //递归
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        preOrder(root, ans);
        return ans;
    }
    private void preOrder(TreeNode root, List<Integer> ans){
        if (root == null){
            return;
        }
        ans.add(root.val);
        preOrder(root.left, ans);
        preOrder(root.right, ans);
    }
    //非递归
    public List<Integer> preorderTraversal2(TreeNode root){
        List<Integer> ans = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null){
            if (p != null){
                ans.add(p.val);
                stack.push(p);
                p = p.left;
            }else {
                p = stack.pop().right;
            }
        }
        return ans;
    }

    /**
     * 145. 二叉树的后序遍历
     * 给你一棵二叉树的根节点 root ，返回其节点值的 后序遍历 。
     */
    //递归
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        postTraversal(root, ans);
        return ans;
    }
    private void postTraversal(TreeNode root, List<Integer> ans){
        if (root == null){
            return;
        }
        postTraversal(root.left, ans);
        postTraversal(root.right, ans);
        ans.add(root.val);
    }

    //二叉树的递归（前、中、后）统一迭代法
    //前序遍历
    public List<Integer> preorderTraversalDDF(TreeNode root) {
        List<Integer> res = new LinkedList<>();
        if (root == null){
            return res;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            if (stack.peek() != null){
                TreeNode node = stack.pop();
                if (node.right != null){
                    stack.push(node.right);
                }
                if (node.left != null){
                    stack.push(node.left);
                }
                stack.push(node);
                stack.push(null);
            }else {
                stack.pop();
                res.add(stack.pop().val);
            }
        }
        return res;
    }

    //中序遍历
    public List<Integer> inorderTraversalDDF(TreeNode root) {
        List<Integer> res = new LinkedList<>();
        if (root == null){
            return res;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            if (stack.peek() != null){
                TreeNode node = stack.pop();
                if (node.right != null){
                    stack.push(node.right);
                }
                stack.push(node);
                stack.push(null);
                if (node.left != null){
                    stack.push(node.left);
                }
            }else {
                stack.pop();
                res.add(stack.pop().val);
            }
        }
        return res;
    }

    //后序遍历
    public List<Integer> postorderTraversalDDF(TreeNode root) {
        List<Integer> res = new LinkedList<>();
        if (root == null){
            return res;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            if (stack.peek() != null){
                TreeNode node = stack.pop();
                stack.push(node);
                stack.push(null);
                if (node.right != null){
                    stack.push(node.right);
                }
                if (node.left != null){
                    stack.push(node.left);
                }
            }else {
                stack.pop();
                res.add(stack.pop().val);
            }
        }
        return res;
    }

    /**
     * 149. 直线上最多的点数
     * 给你一个数组 points ，其中 points[i] = [xi, yi] 表示 X-Y 平面上的一个点。求最多有多少个点在同一条直线上。
     */
    public int maxPoints(int[][] points) {
        int n = points.length;
        if (n <= 2) {
            return n;
        }
        int ret = 0;
        for (int i = 0; i < n; i++) {
            if (ret >= n - i || ret > n / 2) {
                break;
            }
            Map<Integer, Integer> map = new HashMap<Integer, Integer>();
            for (int j = i + 1; j < n; j++) {
                int x = points[i][0] - points[j][0];
                int y = points[i][1] - points[j][1];
                if (x == 0) {
                    y = 1;
                } else if (y == 0) {
                    x = 1;
                } else {
                    if (y < 0) {
                        x = -x;
                        y = -y;
                    }
                    int gcdXY = gcd(Math.abs(x), Math.abs(y));
                    x /= gcdXY;
                    y /= gcdXY;
                }
                int key = y + x * 20001;
                map.put(key, map.getOrDefault(key, 0) + 1);
            }
            int maxn = 0;
            for (Map.Entry<Integer, Integer> entry: map.entrySet()) {
                int num = entry.getValue();
                maxn = Math.max(maxn, num + 1);
            }
            ret = Math.max(ret, maxn);
        }
        return ret;
    }

    public int gcd(int a, int b) {
        return b != 0 ? gcd(b, a % b) : a;
    }

    //朴素解法（枚举直线 + 枚举统计）
    public int maxPoints2(int[][] points){
        int ans = 1;
        for (int i = 0; i < points.length; i++) {
            int[] x = points[i];
            for (int j = i + 1; j < points.length; j++) {
                int[] y = points[j];
                int cnt = 2;
                for (int k = j + 1; k < points.length; k++) {
                    int[] z = points[k];
                    int s1 = (y[1] - x[1]) * (z[0] - y[0]);
                    int s2 = (z[1] - y[1]) * (y[0] - x[0]);
                    if (s1 == s2){
                        cnt++;
                    }
                }
                ans = Math.max(ans, cnt);
            }
        }
        return ans;
    }

    //优化（枚举直线 + 哈希表统计）
    //在使用「哈希表」进行保存时，为了避免精度问题，我们直接使用字符串进行保存，同时需要将 斜率 约干净
    public int maxPoints3(int[][] points){
        int ans = 1;
        for (int i = 0; i < points.length; i++) {
            Map<String, Integer> map = new HashMap<>();
            // 由当前点 i 出发的直线所经过的最多点数量
            int max = 0;
            for (int j = i + 1; j < points.length; j++) {
                int x1 = points[i][0], y1 = points[i][1];
                int x2 = points[j][0], y2 = points[j][1];
                int a = x1 - x2, b = y1 - y2;
                int k = gcd3(a, b);
                String key = (a / k) + "_" + (b / k);
                map.put(key, map.getOrDefault(key, 0) + 1);
                max = Math.max(max, map.get(key));
            }
            ans = Math.max(ans, max + 1);
        }
        return ans;
    }

    private int gcd3(int a, int b) {
        return b == 0 ? a : gcd3(b, a % b);
    }

    /**
     * 150. 逆波兰表达式求值
     * 根据 逆波兰表示法，求表达式的值。
     *
     * 有效的算符包括+、-、*、/。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
     *
     * 注意两个整数之间的除法只保留整数部分。
     *
     * 可以保证给定的逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
     */
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        Set<String> set = new HashSet<>();
        set.add("+");
        set.add("-");
        set.add("*");
        set.add("/");
        int res;
        for (String token : tokens) {
            if (set.contains(token)){
                int b = stack.pop();
                int a = stack.pop();
                //老版本jdk中，String类只能使用.equals()
                if (token.equals("+")){
                    res = a + b;
                }else if (token.equals("-")){
                    res = a - b;
                }else if (token.equals("*")){
                    res = a * b;
                }else {
                    res = a / b;
                }
                stack.push(res);
            }else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }

    /**
     * 151. 颠倒字符串中的单词
     * 给你一个字符串 s ，颠倒字符串中 单词 的顺序。
     *
     * 单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。
     *
     * 返回 单词 顺序颠倒且 单词 之间用单个空格连接的结果字符串。
     *
     * 注意：输入字符串 s中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。
     */
    public String reverseWords151(String s) {
        StringBuilder sb = new StringBuilder();
        String str = s.trim();
        char[] chars = str.toCharArray();
        int rear = chars.length;
        for (int i = rear - 1; i >= 0; i--) {
            if (chars[i] == ' '){
                sb.append(str.substring(i + 1, rear)).append(" ");
                while (chars[i] == ' '){
                    i--;
                }
                rear = ++i;
            }
        }
        sb.append(str.substring(0, rear));
        return new String(sb);
    }


    //搞错了
    public String reverseWords151_2(String s) {
        int pre = 0, rear = 0;
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c != ' '){
                rear++;
            }else {
                sb.append(reverse(s, pre, rear - 1));
                sb.append(" ");
                rear++;
                pre = rear;
            }
        }
        sb.append(reverse(s, pre ,rear - 1));
        return new String(sb);
    }

    private String reverse(String s, int pre, int rear) {
        StringBuilder sb = new StringBuilder();
        while (pre <= rear){
            sb.append(s.charAt(rear--));
        }
        return new String(sb);
    }



    /**
     * 153. 寻找旋转排序数组中的最小值
     * 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
     * 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
     * 若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
     * 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
     *
     * 给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
     *
     * 你必须设计一个时间复杂度为O(log n) 的算法解决此问题。
     */
    public int findMin(int[] nums) {
        if (nums == null || nums.length == 0){
            return -1;
        }
        if (nums.length == 1){
            return nums[0];
        }
        int low = 0;
        int high = nums.length - 1;
        while (low < high){
            int mid = low + (high - low) / 2;
            if (nums[mid] < nums[high]){
                high = mid;
            }else {
                low = mid + 1;
            }
        }
        return nums[low];
    }

    /**
     * 155. 最小栈
     * 设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
     *
     * 实现 MinStack 类:
     *
     * MinStack() 初始化堆栈对象。
     * void push(int val) 将元素val推入堆栈。
     * void pop() 删除堆栈顶部的元素。
     * int top() 获取堆栈顶部的元素。
     * int getMin() 获取堆栈中的最小元素。
     */
    Deque<Integer> xStack;
    Deque<Integer> minStack;
    public void MinStack() {
        xStack = new LinkedList<>();
        minStack = new LinkedList<>();
        minStack.push(Integer.MAX_VALUE);
    }

    public void push(int val) {
        xStack.push(val);
        minStack.push(Math.min(minStack.peek(), val));
    }

    public void pop() {
        xStack.pop();
        minStack.pop();
    }

    public int top() {
        return xStack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }

    /**
     * 160. 相交链表
     * 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
     */
    public ListNode getIntersectionNode160(ListNode headA, ListNode headB) {
        if (headA == null || headB == null){
            return null;
        }
        ListNode p = headA;
        ListNode q = headB;
        int a = 0;
        int b = 0;
        while (p != null){
            a++;
            p = p.next;
        }
        while (q != null){
            b++;
            q = q.next;
        }
        p = headA;
        q = headB;
        while (a > b){
            p = p.next;
            a--;
        }
        while (a < b){
            q = q.next;
            b--;
        }
        while (p != null){
            if (p.equals(q)){
                return p;
            }
            p = p.next;
            q = q.next;
        }
        return null;
    }
    public ListNode getIntersectionNode160_II(ListNode headA, ListNode headB){
        Set<ListNode> set = new HashSet<>();
        ListNode p = headA;
        while (p != null){
            set.add(p);
            p = p.next;
        }
        p = headB;
        while (p != null){
            if (set.contains(p)){
                return p;
            }
            p = p.next;
        }
        return null;
    }

    /**
     * 162: 寻找峰值
     * 峰值元素是指其值严格大于左右相邻值的元素。
     * <p>
     * 给你一个整数数组nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。
     * <p>
     * 你可以假设nums[-1] = nums[n] = -∞ 。
     * <p>
     * 你必须实现时间复杂度为 O(log n) 的算法来解决此问题。
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：nums = [1,2,3,1]
     * 输出：2
     * 解释：3 是峰值元素，你的函数应该返回其索引 2。
     * 示例2：
     * <p>
     * 输入：nums = [1,2,1,3,5,6,4]
     * 输出：1 或 5
     * 解释：你的函数可以返回索引 1，其峰值元素为 2；
     *    或者返回索引 5， 其峰值元素为 6。
     */
    public int findPeakElement(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = (l + r) / 2;
            if (nums[mid] > nums[mid + 1]) r = mid;
            else l = mid + 1;
        }
        return r;
    }

    public int findPeakElement2(int[] nums) {
        int low = 0;
        int high = nums.length - 1;
        while (low < high){
            int mid = low + (high - low) / 2;
            if (nums[mid] > nums[mid + 1]){
                high = mid;
            }else {
                low = mid + 1;
            }
        }
        return high;
    }

    /**
     * 167. 两数之和 II - 输入有序数组
     * 给你一个下标从 1 开始的整数数组numbers ，该数组已按 非递减顺序排列 ，请你从数组中找出满足相加之和等于目标数target 的两个数。
     * 如果设这两个数分别是 numbers[index1] 和 numbers[index2] ，则 1 <= index1 < index2 <= numbers.length 。
     *
     * 以长度为 2 的整数数组 [index1, index2] 的形式返回这两个整数的下标 index1 和 index2。
     *
     * 你可以假设每个输入 只对应唯一的答案 ，而且你 不可以 重复使用相同的元素。
     *
     * 你所设计的解决方案必须只使用常量级的额外空间。
     */
    public int[] twoSum_167(int[] numbers, int target) {
        int i = 0;
        int j = numbers.length - 1;
        while (i < j){
            int sum = numbers[i] + numbers[j];
            if (sum < target){
                i++;
            }else if (sum > target){
                j--;
            }else {
                return new int[]{i + 1, j + 1};
            }
        }
        return new int[]{-1, -1};
    }

    /**
     * 168. Excel表列名称
     * 给你一个整数 columnNumber ，返回它在 Excel 表中相对应的列名称。
     */
    public String convertToTitle(int columnNumber) {
        StringBuilder sb = new StringBuilder();
        while (columnNumber > 0){
            int num = (columnNumber - 1) % 26 + 1;
            sb.append((char) (num - 1  + 'A'));
            columnNumber = (columnNumber - num) / 26;
        }
        return sb.reverse().toString();
    }

    /**
     * 169. 多数元素
     * 给定一个大小为 n 的数组nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于⌊ n/2 ⌋的元素。
     *
     * 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
     */
    public int majorityElement169(int[] nums) {
        int count = 0;
        int ans = 0;
        for (int num : nums) {
            if (count == 0){
                ans = num;
                count++;
            }else {
                if (num == ans){
                    count++;
                }else {
                    count--;
                }
            }
        }
        return ans;
    }

    /**
     * 189. 轮转数组
     * 给你一个数组，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
     */
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        rotateHelper(nums, 0, nums.length - k - 1);
        rotateHelper(nums, nums.length - k, nums.length - 1);
        rotateHelper(nums, 0, nums.length - 1);
    }

    private void rotateHelper(int[] nums, int low, int high) {
        while (low < high){
            nums[low] ^= nums[high];
            nums[high] ^= nums[low];
            nums[low++] ^= nums[high--];
        }
    }

    /**
     * 190. 颠倒二进制位
     * 颠倒给定的 32 位无符号整数的二进制位。
     *
     * 提示：
     *
     * 请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
     * 在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在 示例 2中，输入表示有符号整数 -3，输出表示有符号整数 -1073741825。
     */
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        return Integer.reverse(n);
    }

    //位运算分治
    public int reverseBits2(int n) {
        final int M1 = 0x55555555;
        final int M2 = 0x33333333;
        final int M3 = 0x0f0f0f0f;
        final int M4 = 0x00ff00ff;
        n = n >>> 1 & M1 | (n & M1) << 1;
        n = n >>> 2 & M2 | (n & M2) << 2;
        n = n >>> 4 & M3 | (n & M3) << 4;
        n = n >>> 8 & M4 | (n & M4) << 8;
        return n >>> 16 | n << 16;
    }

    /**
     * 191. 位1的个数
     * 编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。
     */
    // you need to treat n as an unsigned value
    public int hammingWeight191(int n) {
        int ret = 0;
        while (n != 0){
            n &= n - 1;
            ret++;
        }
        return ret;
    }

    /**
     * 198. 打家劫舍
     * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
     * 如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     *
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
     */
    //动态规划
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0){
            return 0;
        }
        if (nums.length == 1){
            return nums[0];
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[nums.length - 1];
    }
    //动态规划+滚动数组
    public int rob2(int[] nums) {
        if (nums == null || nums.length == 0){
            return 0;
        }
        if (nums.length == 1){
            return nums[0];
        }
        int first = nums[0], second = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            int temp = second;
            second = Math.max(first + nums[i], second);
            first = temp;
        }
        return second;
    }

    /**
     * 199. 二叉树的右视图
     * 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
     */
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new LinkedList<>();
        if (root == null){
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int len = queue.size();
            for (int i = 0; i < len; i++) {
                TreeNode node = queue.poll();
                if (i == len - 1){
                    res.add(node.val);
                }
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
        }
        return res;
    }

    /**
     * 200. 岛屿数量
     * 给你一个由'1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     *
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     *
     * 此外，你可以假设该网格的四条边均被水包围。
     */
    public int numIslands(char[][] grid) {
        int res = 0;
        Stack<int[]> stack = new Stack<>();
        int[] directions = {1, 0, -1, 0, 1};
        for (int row = 0; row < grid.length; row++) {
            for (int col = 0; col < grid[0].length; col++) {
                if (grid[row][col] == '1'){
                    stack.push(new int[]{row, col});
                    dfs(grid, stack, directions);
                    res++;
                }
            }
        }
        return res;
    }

    private void dfs(char[][] grid, Stack<int[]> stack, int[] directions){
        while (!stack.isEmpty()){
            int[] local = stack.pop();
            int row = local[0];
            int col = local[1];
            if (grid[row][col] == '1'){
                grid[row][col] = '0';
                for (int i = 0; i < directions.length - 1; i++) {
                    int nRow = row + directions[i];
                    int nCol = col + directions[i + 1];
                    if (nRow < 0 || nCol < 0 || nRow >= grid.length || nCol >= grid[0].length){
                        continue;
                    }else if (grid[nRow][nCol] == '1'){
                        stack.push(new int[]{nRow, nCol});
                    }
                }
            }
        }
    }

    public int numIslands2(char[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1'){
                    dfs(grid, i, j);
                    res++;
                }
            }
        }
        return res;
    }

    private void dfs(char[][] grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == '0') {
            return;
        }
        grid[i][j] = '0';
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }

    /**
     * 201. 数字范围按位与
     * 给你两个整数 left 和 right ，表示区间 [left, right] ，返回此区间内所有数字 按位与 的结果（包含 left 、right 端点）。
     */
    //Brian Kernighan 算法
    public int rangeBitwiseAnd(int left, int right) {
        while (left < right){
            right &= (right - 1);
        }
        return right;
    }

    public int rangeBitwiseAnd2(int left, int right) {
        int shift = 0;
        while (left < right){
            left >>= 1;
            right >>= 1;
            shift++;
        }
        return right <<= shift;
    }

    /**
     * 202. 快乐数
     * 编写一个算法来判断一个数 n 是不是快乐数。
     *
     * 「快乐数」定义为：
     *
     * 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
     * 然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
     * 如果这个过程 结果为1，那么这个数就是快乐数。
     * 如果 n 是 快乐数 就返回 true ；不是，则返回 false 。
     */
    Set<Integer> numIsHappy = new HashSet<>();
    public boolean isHappy(int n) {
        if (n == 1){
            return true;
        }
        int ans = 0;
        while (n > 0){
            ans += Math.pow(n % 10, 2);
            n /= 10;
        }
        if (ans == 1){
            return true;
        }else if (numIsHappy.add(ans)){
            return isHappy(ans);
        }else {
            return false;
        }
    }

    public boolean isHappy2(int n) {
        if (numIsHappy.contains(n)){
            return false;
        }
        numIsHappy.add(n);
        int temp = 0;
        while (n > 0){
            temp += Math.pow(n % 10, 2);
            n /= 10;
        }
        if (temp == 1){
            return true;
        }
        return isHappy2(temp);
    }

    public boolean isHappy3(int n) {
        if (n == 1){
            return true;
        }
        if (numIsHappy.contains(n)){
            return false;
        }
        numIsHappy.add(n);
        int sum = 0;
        while (n > 0){
            int m = n % 10;
            sum += m * m;
            n /= 10;
        }
        return isHappy3(sum);
    }

    /**
     * 203. 移除链表元素
     * 给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。
     */
    public ListNode removeElements(ListNode head, int val) {
        ListNode p = new ListNode();
        p.next = head;
        head = p;
        while (p.next != null){
            if (p.next.val == val){
                p.next = p.next.next;
            }else {
                p = p.next;
            }
        }
        return head.next;
    }

    /**
     * 206. 反转链表
     * 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
     */
    public ListNode reverseList206(ListNode head) {
        ListNode p = new ListNode();
        p.next = head;
        head = p;
        p = p.next;
        head.next = null;
        ListNode q = p;
        while (p != null){
            q = p.next;
            p.next = head.next;
            head.next = p;
            p = q;
        }
        return head.next;
    }

    /**
     * 209. 长度最小的子数组
     * 给定一个含有n个正整数的数组和一个正整数 target 。
     *
     * 找出该数组中满足其和 ≥ target 的长度最小的 连续子数组[numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
     */
    public int minSubArrayLen(int target, int[] nums) {
        int minLen = nums.length;
        int left = 0, right = 0;
        int sum = 0;
        while (right < nums.length){
            sum += nums[right];
            while (sum >= target){
                minLen = Math.min(minLen, right - left + 1);
                sum -= nums[left];
                left++;
            }
            right++;
        }
        return left > 0 ? minLen : 0;
    }

    /**
     * 212. 单词搜索 II
     * 给定一个m x n 二维字符网格board和一个单词（字符串）列表 words，返回所有二维网格上的单词。
     *
     * 单词必须按照字母顺序，通过 相邻的单元格 内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。
     */
    int[][] dirs = {{1,0}, {-1, 0}, {0, 1}, {0, -1}};
    public List<String> findWords(char[][] board, String[] words) {
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        Set<String> ans = new HashSet<>();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                dfs(board, trie, i, j, ans);
            }
        }
        return  new ArrayList<>(ans);
    }
    public void dfs(char[][] board, Trie now, int i, int j, Set<String> ans){
        if (!now.children.containsKey(board[i][j])){
            return;
        }
        char ch = board[i][j];
        now = now.children.get(ch);
        if (!"".equals(now.word)){
            ans.add(now.word);
        }
        board[i][j] = '#';
        for (int[] dir : dirs) {
            int m = i + dir[0];
            int n = j + dir[1];
            if (m >= 0 && m < board.length && n >= 0 && n < board[0].length){
                dfs(board, now, m, n , ans);
            }
        }
        board[i][j] = ch;
    }
    class Trie {
        String word;
        Map<Character, Trie> children;
        boolean isWord;

        public Trie() {
            this.word = "";
            this.children = new HashMap<Character, Trie>();
        }

        public void insert(String word) {
            Trie cur = this;
            for (int i = 0; i < word.length(); ++i) {
                char c = word.charAt(i);
                if (!cur.children.containsKey(c)) {
                    cur.children.put(c, new Trie());
                }
                cur = cur.children.get(c);
            }
            cur.word = word;
        }
    }


    /**
     * 213. 打家劫舍 II
     * 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。
     * 同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
     *
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。
     */
    public int rob213(int[] nums) {
        int len = nums.length;
        if (nums == null || len == 0){
            return 0;
        }
        if (len == 1){
            return nums[0];
        }
        return Math.max(myRob(Arrays.copyOfRange(nums, 0, nums.length - 1)), myRob(Arrays.copyOfRange(nums, 1, nums.length)));
    }

    private int myRob(int[] nums) {
        int pre = 0, cur = 0, temp;
        for (int num : nums) {
            temp = cur;
            cur = Math.max(pre + num, cur);
            pre = temp;
        }
        return cur;
    }


    /**
     * 215. 数组中的第K个最大元素
     * 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
     *
     * 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
     */
    public int findKthLargest(int[] nums, int k) {
        return partition(nums, 0, nums.length - 1, nums.length - k);
    }
    private int partition(int[] arr, int left, int right, int k){
        int i = left;
        int j = right;
        int pivot = arr[i];
        while (i < j){
            while (i < j && arr[j] > pivot){
                j--;
            }
            if (i < j){
                arr[i] = arr[j];
                i++;
            }
            while (i < j && arr[i] < pivot){
                i++;
            }
            if (i < j){
                arr[j] = arr[i];
                j--;
            }
            arr[i] = pivot;
        }
        if (i == k){
            return arr[i];
        }else if (i < k){
            return partition(arr, i + 1, right, k);
        }else {
            return partition(arr, left, i - 1, k);
        }
    }

    //快速查找
    Random Rand = new Random();
    public int findKthLargest_2(int[] nums, int k) {
        int len = nums.length - 1;
        quickSort(nums, 0, len, k);
        return nums[k - 1];
    }

    private void quickSort(int[] nums, int left, int right, int k) {
        if (left >= right){
            return;
        }
        int pos = Rand.nextInt(right - left + 1) + left;
        int temp = nums[left];
        swap(nums, pos, right);
        int index = left;
        for (int i = left; i < right; i++) {
            if (nums[i] >= temp){
                swap(nums, index++, i);
            }
        }
        swap(nums, right, index);
        if (index == k - 1){
            return;
        }else if (index < k - 1){
            quickSort(nums, index + 1, right, k);
        }else {
            quickSort(nums, left, index - 1, k);
        }
    }
    private void swap(int[] nums, int left, int right) {
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
    }

    /**
     * 225. 用队列实现栈
     */

    /**
     * 231. 2 的幂
     * 给你一个整数 n，请你判断该整数是否是 2 的幂次方。如果是，返回 true ；否则，返回 false 。
     *
     * 如果存在一个整数 x 使得n == 2x ，则认为 n 是 2 的幂次方。
     */
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    /**
     * 232. 用栈实现队列
     */

    /**
     * 239. 滑动窗口最大值
     * 给你一个整数数组 nums，有一个大小为k的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k个数字。滑动窗口每次只向右移动一位。
     *
     * 返回 滑动窗口中的最大值 。
     */
    public int[] maxSlidingWindow239(int[] nums, int k) {
        if (nums.length == 1){
            return nums;
        }
        int[] res = new int[nums.length - k + 1];
        int num = 0;
        MyOrderQueue orderQueue = new MyOrderQueue();
        for (int i = 0; i < k; i++) {
            orderQueue.add(nums[i]);
        }
        res[num++] = orderQueue.peek();
        for (int i = k; i < nums.length; i++) {
            orderQueue.poll(nums[i - k]);
            orderQueue.add(nums[i]);
            res[num++] = orderQueue.peek();
        }
        return res;
    }

    /**
     * 242. 有效的字母异位词
     * 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
     *
     * 注意：若s 和 t中每个字符出现的次数都相同，则称s 和 t互为字母异位词。
     */
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()){
            return false;
        }
        int[] strs = new int[26];
        char[] arr1 = s.toCharArray();
        char[] arr2 = t.toCharArray();
        for (char c : arr1) {
            strs[c - 97]++;
        }
        for (char c : arr2) {
            strs[c - 97]--;
            if (strs[c - 97] < 0){
                return false;
            }
        }
        return true;
    }

    public boolean isAnagram2(String s, String t) {
        if (s.length() != t.length()){
            return false;
        }
        int[] flag = new int[26];
        for (char c : t.toCharArray()) {
            flag[c - 'a']++;
        }
        for (char c : s.toCharArray()) {
            flag[c - 'a']--;
            if (flag[c - 'a']  < 0){
                return false;
            }
        }
        return true;
    }


    /**
     * 278. 第一个错误的版本
     * 你是产品经理，目前正在带领一个团队开发新的产品。不幸的是，你的产品的最新版本没有通过质量检测。由于每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本都是错的。
     *
     * 假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。
     *
     * 你可以通过调用bool isBadVersion(version)接口来判断版本号 version 是否在单元测试中出错。实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。
     */
    public int firstBadVersion(int n) {
        int low = 1;
        int high = n;
        while (low < high){
            int mid = low + (high - low) / 2;
            if (isBadVersion(mid)){
                high = mid;
            }else {
                low = mid + 1;
            }
        }
        return high;
    }

    private boolean isBadVersion(int n) {
        if (n >= 1702766719){
            return true;
        }else {
            return false;
        }
    }

    /**
     * 283. 移动零
     * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
     *
     * 请注意 ，必须在不复制数组的情况下原地对数组进行操作。
     */
    public void moveZeroes(int[] nums) {
        int p = 0;
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] == 0 && i < nums.length - 1){
                i++;
            }
            nums[p++] = nums[i];
        }
        for (int i = p; i < nums.length; i++) {
            nums[i] = 0;
        }
    }

    /**
     * 300. 最长递增子序列
     * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
     *
     * 子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
     */
    public int lengthOfLIS(int[] nums) {
        if (nums.length == 0){
            return 0;
        }
        int n = nums.length, res = 0;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]){
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    public int lengthOfLIS2(int[] nums) {
        /*
        int[] tails = new int[nums.length];
        int res = 0;
        for (int num : nums) {
            int i = 0, j = res;
            while (i < j){
                int mid = (i + j) / 2;
                if (tails[mid] < num){
                    i = mid + 1;
                }else {
                    j = mid;
                }
                tails[i] = num;
                if (res == j){
                    res++;
                }
            }
        }
        return res;
         */
        int[] tails = new int[nums.length];
        int res = 0;
        for (int num : nums) {
           int i = 0, j = res;
           while (i < j){
               int m = (i + j) / 2;
               if (tails[m] < num){
                   i = m + 1;
               }else {
                   j = m;
               }
           }
            tails[i] = num;
            if (j == res){
                res++;
            }
        }
        return res;
    }


    /**
     * 322. 零钱兑换
     * 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
     *
     * 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回-1 。
     *
     * 你可以认为每种硬币的数量是无限的。
     */
    public int coinChange(int[] coins, int amount) {
        int max = Integer.MAX_VALUE;
        int[] dp = new int[amount + 1];
        for (int i = 1; i < dp.length; i++) {
            dp[i] = max;
        }
        dp[0] = 0;
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                if (dp[i - coin] != max){
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == max ? -1 : dp[amount];
    }

    /**
     * 343. 整数拆分
     * 给定一个正整数 n ，将其拆分为 k 个 正整数 的和（ k >= 2 ），并使这些整数的乘积最大化。
     *
     * 返回 你可以获得的最大乘积 。
     */
    public int integerBreak(int n) {
        if (n <= 2){
            return 1;
        }
        if (n == 3){
            return 2;
        }
        int res = 1;
        while (n > 4){
            res *= 3;
            n -= 3;
        }
        return res * n;
    }

    public int integerBreak2(int n) {
        int[] dp = new int[n + 1];
        for (int i = 2; i <= n; i++) {
            int curMax = 0;
            for (int j = 1; j < i; j++) {
                curMax = Math.max(curMax, Math.max(j * (i - j), j * dp[i - j]));
            }
            dp[i] = curMax;
        }
        return dp[n];
    }


    /**
     * 344. 反转字符串
     * 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。
     *
     * 不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
     */
    public void reverseString(char[] s) {
        int i = 0;
        int j = s.length - 1;
        while (i < j){
            s[i] ^= s[j];
            s[j] ^= s[i];
            s[i++] ^= s[j--];
        }
    }

    /**
     * 347. 前 K 个高频元素
     * 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
     */
    public int[] topKFrequent(int[] nums, int k) {
        if (k == nums.length){
            return nums;
        }else if (k > nums.length){
            return new int[0];
        }
        int[] ans = new int[k];
        Map<Integer, Integer> map = new HashMap<>();
        PriorityQueue<Integer> queue = new PriorityQueue<>(((o1, o2) -> map.get(o2) - map.get(o1)));
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        for (Integer it : map.keySet()) {
            queue.offer(it);
        }
        for (int i = 0; i < k; i++) {
            ans[i] = queue.poll();
        }
        return ans;
    }

    public int[] topKFrequent2(int[] nums, int k) {
        int[] res = new int[k];
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        Set<Map.Entry<Integer, Integer>> entries = map.entrySet();
        PriorityQueue<Map.Entry<Integer, Integer>> queue = new PriorityQueue<>(((o1, o2) -> o1.getValue() - o2.getValue()));
        for (Map.Entry<Integer, Integer> entry : entries) {
            queue.offer(entry);
            if (queue.size() > k){
                queue.poll();
            }
        }
        for (int i = k - 1; i >= 0 ; i--) {
            res[i] = queue.poll().getKey();
        }
        return res;
    }

    /**
     * 349. 两个数组的交集
     * 给定两个数组 nums1 和 nums2 ，返回 它们的交集 。输出结果中的每个元素一定是 唯一 的。我们可以 不考虑输出结果的顺序 。
     */
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> list = new HashSet<>();
        Set<Integer> set = new HashSet<>();
        for (int i : nums1) {
            set.add(i);
        }
        for (int i : nums2) {
            if (set.contains(i)){
                list.add(i);
            }
        }
        int[] res = new int[list.size()];
        int index = 0;
        for (int i : list) {
            res[index++] = i;
        }
        return res;
    }

    /**
     * 383. 赎金信
     * 给你两个字符串：ransomNote 和 magazine ，判断 ransomNote 能不能由 magazine 里面的字符构成。
     *
     * 如果可以，返回 true ；否则返回 false 。
     *
     * magazine 中的每个字符只能在 ransomNote 中使用一次。
     */
    public boolean canConstruct(String ransomNote, String magazine) {
        int[] book = new int[26];
        for (char c : magazine.toCharArray()) {
            book[c - 'a']++;
        }
        for (char c : ransomNote.toCharArray()) {
            book[c - 'a']--;
            if (book[c - 'a'] < 0){
                return false;
            }
        }
        return true;
    }

    /**
     * 413. 等差数列划分
     * 如果一个数列 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该数列为等差数列。
     *
     * 例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。
     * 给你一个整数数组 nums ，返回数组 nums 中所有为等差数组的 子数组 个数。
     *
     * 子数组 是数组中的一个连续序列。
     */
    public int numberOfArithmeticSlices(int[] nums) {
        if (nums == null || nums.length < 3){
            return 0;
        }
        int temp = nums[1] - nums[0], len = 0, res = 0;
        for (int i = 2; i < nums.length; i++) {
            if (nums[i] - nums[i - 1] == temp){
                len++;
            }else {
                temp = nums[i] - nums[i - 1];
                len = 0;
            }
            res += len;
        }
        return res;
    }

    /**
     * 416. 分割等和子集
     * 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
     */
    //超时
    public boolean canPartition(int[] nums) {
        if (nums.length < 2){
            return false;
        }
        Arrays.sort(nums);
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        int low = 0;
        int high = nums.length - 1;
        while (low < high){
            int temp = nums[low];
            nums[low++] = nums[high];
            nums[high--] = temp;
        }
        if (sum % 2 != 0){
            return false;
        }
        int target = sum / 2;
        int[] bucket = new int[3];
        return backtrack(nums, bucket, 0, 2, target);
    }

    private boolean backtrack(int[] nums, int[] bucket, int index, int k, int target) {
        if (index == nums.length){
            return true;
        }
        for (int i = 0; i < k; i++) {
            if (i > 0 && bucket[i] == bucket[i - 1]){
                continue;
            }
            if (bucket[i] + nums[index] > target){
                continue;
            }
            bucket[i] += nums[index];
            if (backtrack(nums, bucket, index + 1, k, target)) {
                return true;
            }
            bucket[i] -= nums[index];
        }
        return false;
    }

    //转化为背包问题
    public boolean canPartition2(int[] nums) {
        if (nums.length < 2){
            return false;
        }
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum % 2 != 0){
            return false;
        }
        int target = sum / 2;
        int[] dp = new int[target + 1];


        for (int i = 0; i < nums.length; i++) {
            for (int j = target; j >= nums[i] ; j--) {
                dp[j] = Math.max(dp[j], dp[j - nums[i]] + nums[i]);
            }
        }
        return dp[target] == target;
    }

    /**
     * 438: 找出字符串中所有字母异位词
     * 给定两个字符串s和 p，找到s中所有p的异位词的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
     * <p>
     * 异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
     */
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> list = new ArrayList<>();
        if (s.length() < p.length()) {
            return list;
        }
        int[] sCount = new int[26];
        int[] pCount = new int[26];

        for (int i = 0; i < p.length(); ++i) {
            ++sCount[s.charAt(i) - 'a'];
            ++pCount[p.charAt(i) - 'a'];
        }

        if (Arrays.equals(sCount, pCount)) {
            list.add(0);
        }
        for (int i = 0; i < s.length() - p.length(); i++) {
            --sCount[s.charAt(i) - 'a'];
            ++sCount[s.charAt(i + p.length()) - 'a'];
            if (Arrays.equals(sCount, pCount)) {
                list.add(i + 1);
            }
        }
        return list;
    }

    //滑动窗口
    public List<Integer> findAnagrams2(String s, String p) {
        List<Integer> res = new ArrayList<>();
        if (s.length() < p.length()){
            return res;
        }
        int[] pCnt = new int[26];
        for (int i = 0; i < p.length(); i++) {
            pCnt[p.charAt(i) - 'a']++;
        }
        int left = 0;
        for (int right = 0; right < s.length(); right++) {
            int curR = s.charAt(right) - 'a';
            pCnt[curR]--;
            while (pCnt[curR] < 0){
                pCnt[s.charAt(left) - 'a']++;
                left++;
            }
            if (right - left + 1 == p.length()){
                res.add(left);
            }
        }
        return res;
    }

    /**
     * 454. 四数相加 II
     * 给你四个整数数组 nums1、nums2、nums3 和 nums4 ，数组长度都是 n ，请你计算有多少个元组 (i, j, k, l) 能满足：
     *
     * 0 <= i, j, k, l < n
     * nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
     */
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        //定义 一个map，key放a和b两数之和，value 放a和b两数之和出现的次数
        Map<Integer, Integer> map = new HashMap<>();
        //遍历大A和大B数组，统计两个数组元素之和，和出现的次数，放到map中
        for (int i : nums1) {
            for (int j : nums2) {
                int temp = i + j;
                if (map.containsKey(temp)){
                    map.replace(temp, map.get(temp) + 1);
                }else {
                    map.put(temp, 1);
                }
            }
        }
        int res = 0;
        //统计剩余的两个元素的和，在map中找到是否存在相加为0的情况，同时记录
        for (int i : nums3) {
            for (int j : nums4) {
                int temp = i + j;
                if (map.containsKey(0 - temp)){
                    res += map.get(0 - temp);
                }
            }
        }
        return res;
    }

    /**
     * 459. 重复的子字符串
     * 给定一个非空的字符串 s ，检查是否可以通过由它的一个子串重复多次构成。
     */
    //KMP算法
    public boolean repeatedSubstringPattern(String s) {
        int n = s.length();
        int[] next = new int[n];
        next[0] = 0;
        int j = 0;
        for (int i = 1; i < n; i++) {
            while (j > 0 && s.charAt(i) != s.charAt(j)){
                j = next[j - 1];
            }
            if (s.charAt(i) == s.charAt(j)){
                j++;
            }
            next[i] = j;
        }
        int len = 0;
        for (int i : next) {
            if (i == 0){
                len++;
            }else {
                break;
            }
        }
        return  next[n - 1] != 0 && n % (n - next[n - 1]) == 0;
    }

    /**
     * 461. 汉明距离
     * 两个整数之间的 汉明距离 指的是这两个数字对应二进制位不同的位置的数目。
     *
     * 给你两个整数 x 和 y，计算并返回它们之间的汉明距离。
     */
    public int hammingDistance(int x, int y) {
        int diff = x ^ y;
        int res = 0;
        while (diff > 0){
            diff &= (diff - 1);
            res++;
        }
        return res;
    }

    /**
     * 462. 最少移动次数使数组元素相等 II
     * 给你一个长度为 n 的整数数组 nums ，返回使所有数组元素相等需要的最少移动数。
     *
     * 在一步操作中，你可以使数组中的一个元素加 1 或者减 1 。
     */

    public int minMoves2(int[] nums) {
        int count = 0;
        int maxNum = findKthLargest(nums, nums.length / 2 + 1);
        for (int num : nums) {
            count += (Math.abs(maxNum - num));
        }
        return count;
    }

    /**
     * 473. 火柴拼正方形
     * 你将得到一个整数数组 matchsticks ，其中 matchsticks[i] 是第 i个火柴棒的长度。你要用 所有的火柴棍拼成一个正方形。
     * 你 不能折断 任何一根火柴棒，但你可以把它们连在一起，而且每根火柴棒必须 使用一次 。
     *
     * 如果你能使这个正方形，则返回 true ，否则返回 false 。
     */
    public boolean makesquare(int[] matchsticks) {
        int k = 4;
        if (matchsticks.length < k){
            return false;
        }
        Arrays.sort(matchsticks);
        int low = 0;
        int high = matchsticks.length - 1;
        while (low < high){
            int temp = matchsticks[low];
            matchsticks[low++] = matchsticks[high];
            matchsticks[high--] = temp;
        }
        int sum = 0;
        for (int stick : matchsticks) {
            sum += stick;
        }
        if (sum % k != 0){
            return false;
        }
        int target = sum / k;
        int[] buckets = new int[k + 1];
        return backtrackSticks(matchsticks, buckets, 0, k, target);
    }
    
    private boolean backtrackSticks(int[] matchsticks, int[] buckets, int index, int k, int target){
        if (index == matchsticks.length){
            return true;
        }
        for (int i = 0; i < k; i++) {
            if (i > 0 && buckets[i] == buckets[i - 1]){
                continue;
            }
            if (buckets[i] + matchsticks[index] > target){
                continue;
            }
            buckets[i] += matchsticks[index];
            if (backtrackSticks(matchsticks, buckets, index + 1, k, target)){
                return true;
            }
            buckets[i] -= matchsticks[index];
        }
        return false;
    }

    /**
     * 474. 一和零
     * 给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
     *
     * 请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。
     *
     * 如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。
     */
    public int findMaxForm(String[] strs, int m, int n) {
        int[][] dp = new int[m + 1][n + 1];
        int oneNum, zeroNum;
        for (String str : strs) {
            oneNum = 0;
            zeroNum = 0;
            for (char ch : str.toCharArray()) {
                if (ch == '0'){
                    zeroNum++;
                }else {
                    oneNum++;
                }
            }
            for (int i = m; i >= zeroNum; i--) {
                for (int j = n; j >= oneNum; j--) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1);
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 494. 目标和
     * 给你一个整数数组 nums 和一个整数 target 。
     *
     * 向数组中的每个整数前添加'+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
     *
     * 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
     * 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
     */
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (Math.abs(target) > sum || (target + sum) % 2 != 0){
            return 0;
        }
        int bagSize = Math.abs((sum + target) / 2);
        int[] dp = new int[bagSize + 1];
        dp[0] = 1;
        for (int num : nums) {
            for (int j = bagSize; j >= num ; j--) {
                dp[j] += dp[j - num];
            }
        }
        return dp[bagSize];
    }

    /**
     * 509. 斐波那契数
     * 斐波那契数（通常用F(n) 表示）形成的序列称为 斐波那契数列 。该数列由0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
     *
     * F(0) = 0，F(1)= 1
     * F(n) = F(n - 1) + F(n - 2)，其中 n > 1
     * 给定n ，请计算 F(n) 。
     */
    public int fib_509(int n) {
        int[] cache = new int[n + 1];
        return fibCalculate(n, cache);
    }

    private int fibCalculate(int n, int[] cache) {
        if (n == 0){
            cache[0] = 0;
            return 0;
        }
        if (n == 1){
            cache[1] = 1;
            return 1;
        }
        if (cache[n] != 0){
            return cache[n];
        }
        cache[n] = fibCalculate(n - 1, cache) + fibCalculate(n - 2, cache);
        return cache[n];
    }

    /**
     * 541. 反转字符串 II
     * 给定一个字符串 s 和一个整数 k，从字符串开头算起，每计数至 2k 个字符，就反转这 2k 字符中的前 k 个字符。
     *
     * 如果剩余字符少于 k 个，则将剩余字符全部反转。
     * 如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
     */
    public String reverseStr(String s, int k) {
        Stack<Character> stack = new Stack<>();
        StringBuilder sb = new StringBuilder();
        int cnt = 1;
        for (char c : s.toCharArray()) {
            if (cnt < k) {
                stack.push(c);
            } else if (cnt == k) {
                sb.append(c);
                while (!stack.isEmpty()) {
                    sb.append(stack.pop());
                }
            } else if (cnt < k * 2) {
                sb.append(c);
            } else {
                sb.append(c);
                cnt = 0;
            }
            cnt++;
        }
        while (!stack.isEmpty()){
            sb.append(stack.pop());
        }
        return sb.toString();
    }

    public String reverseStr2(String s, int k) {
        char[] chars = s.toCharArray();
        for (int i = 0; i < chars.length; i += k * 2) {
            int low = i, high = Math.min(i + k, chars.length) - 1;
            while (low < high){
                chars[low] ^= chars[high];
                chars[high] ^= chars[low];
                chars[low++] ^= chars[high--];
            }
        }
        return new String(chars);
    }

    /**
     * 542. 01 矩阵
     * 给定一个由 0 和 1 组成的矩阵 mat，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。
     *
     * 两个相邻元素间的距离为 1 。
     */
    public int[][] updateMatrix(int[][] mat) {
        Queue<int[]> queue = new LinkedList<>();
        int[] directions = {-1, 0, 1, 0, -1};
        for (int row = 0; row < mat.length; row++) {
            for (int col = 0; col < mat[0].length; col++) {
                if (mat[row][col] == 0){
                    queue.offer(new int[]{row, col});
                }else {
                    //标记非零元素为负，和遍历后设定的正数距离加以区分
                    mat[row][col] = -1;
                }
            }
        }
        int step = 1;
        while (!queue.isEmpty()){
            //对当前队列中所有0元素遍历，所有元素向四周走一步
            int size = queue.size();
            while (size-- > 0){
                //获取队列中的元素位置
                int[] cur = queue.poll();
                //向四个方向依次走一步
                for (int i = 0; i < directions.length - 1; i++) {
                    int x = cur[0] + directions[i];
                    int y = cur[1] + directions[i + 1];
                    //如果超出矩阵范围，或者遇见零元素及设置过距离step的元素则跳过，只对未遍历到的-1操作
                    if (x < 0 || x >= mat.length || y < 0 || y >= mat[0].length || mat[x][y] >= 0){
                        continue;
                    }
                    mat[x][y] = step;
                    queue.offer(new int[]{x, y});
                }
            }
            //下一次遍历到的-1元素相比于前一次step距离+1
            step++;
        }
        return mat;
    }

    /**
     * 547. 省份数量
     * 有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。
     *
     * 省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。
     *
     * 给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。
     *
     * 返回矩阵中 省份 的数量。
     */
    public int findCircleNum(int[][] isConnected) {
        int res = 0;
        int cities = isConnected.length;
        boolean[] visits = new boolean[cities];
        for (int i = 0; i < cities; i++) {
            if (!visits[i]){
                dfs(isConnected, i, visits, cities);
                res++;
            }

        }
        return res;
    }

    private void dfs(int[][] isConnected, int i, boolean[] visited, int cities) {
        for (int j = 0; j < cities; j++) {
            if (isConnected[i][j] == 1 && !visited[j]){
                visited[j] = true;
                dfs(isConnected, j, visited, cities);
            }
        }
    }

    /**
     * 557. 反转字符串中的单词 III
     * 给定一个字符串 s ，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。
     */
    public String reverseWordsIII(String s) {
        StringBuilder revIII = new StringBuilder();
        String[] strs = s.split(" ");
        for (String str : strs) {
            char[] m = str.toCharArray();
            for (int i = m.length - 1; i >= 0; i--) {
                revIII.append(m[i]);
            }
            revIII.append(" ");
        }
        return revIII.deleteCharAt(revIII.length() - 1).toString();
    }

    public String reverseWordsIII_2(String s) {
        StringBuilder sb = new StringBuilder();
        char[] chars = s.toCharArray();
        int tag = 0;
        for (int i = 0; i <= chars.length; i++) {
            if (i == chars.length || chars[i] == ' '){
                for (int j = i - 1; j >= tag; j--) {
                    sb.append(chars[j]);
                }
                sb.append(" ");
                tag = i + 1;
            }
        }
        return sb.deleteCharAt(sb.length() - 1).toString();
    }

    /**
     * 567. 字符串的排列
     * 给你两个字符串s1和s2 ，写一个函数来判断 s2 是否包含 s1的排列。如果是，返回 true ；否则，返回 false 。
     *
     * 换句话说，s1 的排列之一是 s2 的 子串 。
     */
    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length()) {
            return false;
        }
        int[] cnt = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            cnt[s1.charAt(i) - 'a']--;
        }
        int left = 0;
        for (int right = 0; right < s2.length(); right++) {
            int x = s2.charAt(right) - 'a';
            cnt[x]++;
            while (cnt[x] > 0){
                cnt[s2.charAt(left) - 'a']--;
                left++;
            }
            if (right - left + 1 == s1.length()){
                return true;
            }
        }
        return false;
    }

    /**
     * 572. 另一棵树的子树
     * 给你两棵二叉树 root 和 subRoot 。检验 root 中是否包含和 subRoot 具有相同结构和节点值的子树。如果存在，返回 true ；否则，返回 false 。
     *
     * 二叉树 tree 的一棵子树包括 tree 的某个节点和这个节点的所有后代节点。tree 也可以看做它自身的一棵子树。
     */
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null && subRoot == null){
            return true;
        }
        if (root == null || subRoot == null){
            return false;
        }
        return isSameTree572(root, subRoot) || isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
    }

    private boolean isSameTree572(TreeNode root, TreeNode subRoot) {
        if (root == null && subRoot == null){
            return true;
        }
        if (root == null || subRoot == null){
            return false;
        }
        return root.val == subRoot.val && isSameTree572(root.left, subRoot.left) && isSameTree572(root.right, subRoot.right);
    }

    /**
     * 583. 两个字符串的删除操作
     * 给定两个单词 word1 和 word2 ，返回使得 word1 和  word2 相同所需的最小步数。
     *
     * 每步 可以删除任意一个字符串中的一个字符。
     */
    //法一
    public int minDistance(String word1, String word2) {
        int M = word1.length();
        int N = word2.length();
        int[][] dp = new int[M + 1][N + 1];
        for (int i = 1; i <= M; i++) {
            for (int j = 1; j <= N; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return M + N - dp[M][N] * 2;
    }

    //法二： 直接使用动态规划
    public int minDistance2(String word1, String word2) {
        int M = word1.length();
        int N = word2.length();
        int[][] dp = new int[M + 1][N + 1];
        for (int i = 1; i <= M; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= N; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= M; i++) {
            char c1 = word1.charAt(i - 1);
            for (int j = 1; j <= N; j++) {
                if (c1 == word2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1];
                }else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + 1;
                }
            }
        }
        return dp[M][N];
    }


    /**
     * 617. 合并二叉树
     * 给你两棵二叉树： root1 和 root2 。
     *
     * 想象一下，当你将其中一棵覆盖到另一棵之上时，两棵树上的一些节点将会重叠（而另一些不会）。
     * 你需要将这两棵树合并成一棵新二叉树。合并的规则是：如果两个节点重叠，那么将这两个节点的值相加作为合并后节点的新值；否则，不为 null 的节点将直接作为新二叉树的节点。
     *
     * 返回合并后的二叉树。
     *
     * 注意: 合并过程必须从两个树的根节点开始。
     */
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null){
            return root2;
        }
        if (root2 == null){
            return root1;
        }
        return mergeTreesHelper(root1, root2);
    }

    private TreeNode mergeTreesHelper(TreeNode root1, TreeNode root2) {
        TreeNode res = root1;
        root1.val += root2.val;
        if (root1.left != null && root2.left != null){
            mergeTrees(root1.left, root2.left);
        } else if (root1.left == null){
            root1.left = root2.left;
        }
        if (root1.right != null && root2.right != null){
            mergeTrees(root1.right, root2.right);
        } else if (root1.right == null){
            root1.right = root2.right;
        }
        return res;
    }

    /**
     * 637. 二叉树的层平均值
     * 给定一个非空二叉树的根节点 root , 以数组的形式返回每一层节点的平均值。与实际答案相差 10-5 以内的答案可以被接受。
     */
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> res = new LinkedList<>();
        if (root == null){
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int len = queue.size();
            double sum = 0;
            for (int i = 0; i < len; i++) {
                TreeNode node = queue.poll();
                sum += node.val;
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
            res.add(sum / len);
        }
        return res;
    }

    /**
     * 673. 最长递增子序列的个数
     * 给定一个未排序的整数数组 nums ， 返回最长递增子序列的个数 。
     *
     * 注意 这个数列必须是 严格 递增的。
     */
    public int findNumberOfLIS(int[] nums) {
        if (nums.length <= 1){
            return nums.length;
        }
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int[] cnt = new int[nums.length];
        Arrays.fill(cnt, 1);

        int maxCount = 0;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]){
                    if (dp[j] + 1 > dp[i]){
                        dp[i] = dp[j] + 1;
                        cnt[i] = cnt[j];
                    }else if (dp[j] + 1 == dp[i]){
                        cnt[i] += cnt[j];
                    }
                }
                maxCount = Math.max(maxCount, dp[i]);
            }
        }
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            if (maxCount == dp[i]){
                res += cnt[i];
            }
        }
        return res;
    }

    /**
     * 695. 岛屿的最大面积
     * 给你一个大小为 m x n 的二进制矩阵 grid 。
     *
     * 岛屿是由一些相邻的1(代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设grid 的四个边缘都被 0（代表水）包围着。
     *
     * 岛屿的面积是岛上值为 1 的单元格的数目。
     *
     * 计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。
     */
    int[][] grid;
    public int maxAreaOfIsland(int[][] grid) {
        this.grid = grid;
        int maxArea = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (checkIsland(i, j)){
                    maxArea = Math.max(maxArea, countArea(i, j));
                }
            }
        }
        return maxArea;
    }

    private int countArea(int i, int j) {
        if (!checkIsland(i, j)){
            return 0;
        }
        grid[i][j] = 0;
        return 1 + countArea(i - 1, j) + countArea(i + 1, j) + countArea(i, j - 1) + countArea(i, j + 1);
    }

    private boolean checkIsland(int sr, int sc) {
        if (sr < 0 || sr >= grid.length || sc < 0 || sc >= grid[0].length || grid[sr][sc] == 0){
            return false;
        }
        return true;
    }

    /**
     * 698. 划分为k个相等的子集
     * 给定一个整数数组  nums 和一个正整数 k，找出是否有可能把这个数组分成 k 个非空子集，其总和都相等。
     */
    public boolean canPartitionKSubsets(int[] nums, int k) {
        if (nums.length < k){
            return false;
        }
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum % k != 0){
            return false;
        }
        int target = sum / k;
        int[] bucket = new int[k + 1];
        Arrays.sort(nums);
        int low = 0;
        int high = nums.length - 1;
        while (low < high){
            int temp = nums[low];
            nums[low++] = nums[high];
            nums[high--] = temp;
        }
        return backtrack(nums, 0, bucket, k, target);

    }

    /**
     * @param nums                      原始数组
     * @param index                     第index个球开始做选择
     * @param bucket                    桶
     * @param k                         k个非空子集，每个子集的和为target
     * @param target                    目标和
     * @return
     */
    private boolean backtrack(int[] nums, int index, int[] bucket, int k, int target) {
        //结束条件： 已经处理完所有球
        if (index == nums.length){
            return true;
        }
        //nums[index]开始做选择
        for (int i = 0; i < k; i++) {
            //如果当前桶和上一个桶内元素和相等，则跳过
            if (i > 0 && bucket[i] == bucket[i - 1]){
                continue;
            }
            //剪枝： 放入球后和超过target，选择下一个桶
            if (bucket[i] + nums[index] > target){
                continue;
            }
            //做选择： 放入i号桶
            bucket[i] += nums[index];
            //处理下一个球， 即nums[index + 1]
            if (backtrack(nums, index + 1, bucket, k, target)){
                return true;
            }
            //撤销选择，挪出i号桶
            bucket[i] -= nums[index];
        }
        // k个桶都不符合要求
        return false;
    }


    /**
     * 700. 二叉搜索树中的搜索
     * 给定二叉搜索树（BST）的根节点root和一个整数值val。
     *
     * 你需要在 BST 中找到节点值等于val的节点。 返回以该节点为根的子树。 如果节点不存在，则返回null。
     */
    public TreeNode searchBST(TreeNode root, int val) {
        if (root == null){
            return null;
        }else if (root.val == val){
            return root;
        }
        TreeNode nodeL = searchBST(root.left, val);
        return nodeL != null ? nodeL : searchBST(root.right, val);
    }

    /**
     * 704. 二分查找
     * 给定一个n个元素有序的（升序）整型数组nums 和一个目标值target ，写一个函数搜索nums中的 target，如果目标值存在返回下标，否则返回 -1。
     */
    public int search_704(int[] nums, int target) {
        return binarySearch(nums, 0, nums.length - 1, target);
    }
    private int binarySearch(int[] nums, int low, int high, int target){
        if (low > high){
            return -1;
        }
        int mid = (low + high) / 2;
        if (nums[mid] == target){
            return mid;
        }else if (nums[mid] > target){
            return binarySearch(nums, low, mid - 1, target);
        }else {
            return binarySearch(nums, mid + 1, high, target);
        }
    }

    /**
     * 707. 设计链表
     * 设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：val和next。val是当前节点的值，next是指向下一个节点的指针/引用。
     * 如果要使用双向链表，则还需要一个属性prev以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。
     *
     * 在链表类中实现这些功能：
     *
     * get(index)：获取链表中第index个节点的值。如果索引无效，则返回-1。
     * addAtHead(val)：在链表的第一个元素之前添加一个值为val的节点。插入后，新节点将成为链表的第一个节点。
     * addAtTail(val)：将值为val 的节点追加到链表的最后一个元素。
     * addAtIndex(index,val)：在链表中的第index个节点之前添加值为val 的节点。如果index等于链表的长度，则该节点将附加到链表的末尾。
     * 如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。
     * deleteAtIndex(index)：如果索引index 有效，则删除链表中的第index 个节点。
     */
    class MyLinkedList {
        int size;
        ListNode head;
        public MyLinkedList() {
            size = 0;
            head = new ListNode();
        }

        public int get(int index) {
            if (index < 0 || index >= size){
                return -1;
            }
            ListNode p = head;
            for (int i = 0; i <= index; i++) {
                p = p.next;
            }
            return p.val;
        }

        public void addAtHead(int val) {
            addAtIndex(0, val);
        }

        public void addAtTail(int val) {
            addAtIndex(size, val);
        }

        public void addAtIndex(int index, int val) {
            if (index > size){
                return;
            }
            if (index < 0){
                index = 0;
            }
            ListNode node = new ListNode();
            node.val = val;
            ListNode p = head;
            for (int i = 0; i < index; i++) {
                p = p.next;
            }
            node.next = p.next;
            p.next = node;
            size++;
        }

        public void deleteAtIndex(int index) {
            if (index < 0 || index >= size){
                return;
            }
            ListNode p = head;
            for (int i = 0; i < index; i++) {
                p = p.next;
            }
            p.next = p.next.next;
            size--;
        }
    }

    /**
     * 713. 乘积小于 K 的子数组
     * 给你一个整数数组 nums 和一个整数 k ，请你返回子数组内所有元素的乘积严格小于 k 的连续子数组的数目。
     */
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1){
            return 0;
        }
        int left = 0;
        int right = 0;
        int mul = 1;
        int res = 0;
        while (right < nums.length){
            mul *= nums[right];
            while (mul >= k){
                mul /= nums[left];
                left++;
            }
            res += right - left + 1;
            right++;
        }
        return res;
    }

    /**
     * 733. 图像渲染
     * 有一幅以m x n的二维整数数组表示的图画image，其中image[i][j]表示该图画的像素值大小。
     *
     * 你也被给予三个整数 sr , sc 和 newColor 。你应该从像素image[sr][sc]开始对图像进行 上色填充 。
     *
     * 为了完成 上色工作 ，从初始像素开始，记录初始坐标的 上下左右四个方向上 像素值与初始坐标相同的相连像素点，
     * 接着再记录这四个方向上符合条件的像素点与他们对应 四个方向上 像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为newColor。
     *
     * 最后返回 经过上色渲染后的图像。
     */
    int[][] image;
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        this.image = image;
        int oldColor = image[sr][sc];
        floodFillHelper(sr, sc, oldColor, newColor);
        return image;
    }

    private void floodFillHelper(int sr, int sc, int oldColor, int newColor) {
        if (!checkArea(sr, sc, oldColor, newColor)) {
            return;
        }
        image[sr][sc] = newColor;
        floodFillHelper(sr + 1, sc, oldColor, newColor);
        floodFillHelper(sr - 1, sc, oldColor, newColor);
        floodFillHelper(sr, sc + 1, oldColor, newColor);
        floodFillHelper(sr, sc - 1, oldColor, newColor);
    }

    private boolean checkArea(int sr, int sc, int oldColor, int newColor) {
        if (sr < 0 || sr >= image.length || sc < 0 || sc >= image[0].length || image[sr][sc] != oldColor || image[sr][sc] == newColor){
            return false;
        }
        return true;
    }

    /**
     * 746. 使用最小花费爬楼梯
     * 给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。
     *
     * 你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。
     *
     * 请你计算并返回达到楼梯顶部的最低花费。
     */
    public int minCostClimbingStairs(int[] cost) {
        int[] cache = new int[cost.length];
        for (int i = 0; i < cost.length; i++) {
            if (i <= 1){
                cache[i] = cost[i];
            }else {
                cache[i] = Math.min(cache[i - 1], cache[i - 2]) + cost[i];
            }
        }
        return Math.min(cache[cache.length - 1], cache[cache.length - 2]);
    }



    /**
     * 784. 字母大小写全排列
     * 给定一个字符串 s ，通过将字符串 s 中的每个字母转变大小写，我们可以获得一个新的字符串。
     *
     * 返回 所有可能得到的字符串集合 。以 任意顺序 返回输出。
     */
    public List<String> letterCasePermutation(String s) {
        char[] chars = s.toCharArray();
        List<String> res = new ArrayList<>();
        backtrack(res, chars, new StringBuilder(), 0);
        return res;
    }

    private void backtrack(List<String> res, char[] chars, StringBuilder sb, int index) {
        if (sb.length() == chars.length){
            res.add(sb.toString());
            return;
        }
        char temp = chars[index];
        sb.append(temp);
        backtrack(res, chars, sb, index + 1);
        sb.deleteCharAt(sb.length() - 1);
        if (chars[index] >= 'a' && chars[index] <= 'z'){
            temp = (char) (chars[index] + 'A' - 'a');
            sb.append(temp);
            backtrack(res, chars, sb, index + 1);
            sb.deleteCharAt(sb.length() - 1);
        }else if (chars[index] >= 'A' && chars[index] <= 'Z'){
            temp = (char)(chars[index] - 'A' + 'a');
            sb.append(temp);
            backtrack(res, chars, sb, index + 1);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    /**
     * 797. 所有可能的路径
     * 给你一个有n个节点的 有向无环图（DAG），请你找出所有从节点 0到节点 n-1的路径并输出（不要求按特定顺序）
     *
     * graph[i]是一个从节点 i 可以访问的所有节点的列表（即从节点 i 到节点graph[i][j]存在一条有向边）。
     */
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> res = new ArrayList<>();
        Deque<Integer> deque = new ArrayDeque<>();
        deque.add(0);
        dfs(graph, 0, graph.length - 1, res, deque);
        return res;
    }

    private void dfs(int[][] graph, int x, int n, List<List<Integer>> res, Deque<Integer> deque) {
        if (x == n){
            res.add(new ArrayList<>(deque));
            return;
        }
        for (int i : graph[x]) {
            deque.offerLast(i);
            dfs(graph, i, n, res, deque);
            deque.pollLast();
        }
    }


    /**
     * 824 山羊拉丁文
     * 给你一个由若干单词组成的句子 sentence ，单词间由空格分隔。每个单词仅由大写和小写英文字母组成。
     * <p>
     * 请你将句子转换为 “山羊拉丁文（Goat Latin）”（一种类似于 猪拉丁文 - Pig Latin 的虚构语言）。山羊拉丁文的规则如下：
     * <p>
     * 如果单词以元音开头（'a', 'e', 'i', 'o', 'u'），在单词后添加"ma"。
     * 例如，单词 "apple" 变为 "applema" 。
     * 如果单词以辅音字母开头（即，非元音字母），移除第一个字符并将它放到末尾，之后再添加"ma"。
     * 例如，单词 "goat" 变为 "oatgma" 。
     * 根据单词在句子中的索引，在单词最后添加与索引相同数量的字母'a'，索引从 1 开始。
     * 例如，在第一个单词后添加 "a" ，在第二个单词后添加 "aa" ，以此类推。
     * 返回将 sentence 转换为山羊拉丁文后的句子。
     */
    public String toGoatLatin(String sentence) {
        StringBuffer sb = new StringBuffer();
        StringBuilder ma = new StringBuilder("ma");
        String aeiou = "aeiouAEIOU";
        boolean flag = true;

        char p = ' ';

        //遍历整个句子
        for (int i = 0; i < sentence.length(); i++) {
            //判断是否是单词开头
            //判断单词开头是否是元音
            if (aeiou.contains(sentence.subSequence(i, i + 1))) {
                //是元音开头
                ma.append("a");
                flag = false;
            } else {
                //不是元音开头
                flag = true;
                ma.append("a");
                p = sentence.charAt(i);
                i++;
            }
            while (i < sentence.length() && sentence.charAt(i) != 32) {
                sb.append(sentence.charAt(i));
                i++;
            }
            if (flag) {
                sb.append(p);
            }
            sb.append(ma).append(' ');

        }
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    /**
     * 844：比较含退格的字符串
     * 给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 true 。# 代表退格字符。
     * <p>
     * 注意：如果对空文本输入退格字符，文本继续为空。
     */
    public boolean backspaceCompare(String s, String t) {
        return (deleteChat(s, '#')).equals(deleteChat(t, '#'));
    }

    private String deleteChat(String s, char temp) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) != temp) {
                sb.append(s.charAt(i));
            } else {
                if (sb.length() != 0) {
                    sb.deleteCharAt(sb.length() - 1);
                }
            }
        }
        return sb.toString();
    }

    public boolean backspaceCompare2(String s, String t) {
        int i = s.length() - 1;
        int j = t.length() - 1;
        int skipS = 0;
        int skipT = 0;
        while (i >= 0 || j >= 0){
            while (i >= 0){
                if (s.charAt(i) == '#'){
                    skipS++;
                    i--;
                }else if (skipS > 0){
                    skipS--;
                    i--;
                }else {
                    break;
                }
            }
            while (j >= 0){
                if (t.charAt(j) == '#'){
                    skipT++;
                    j--;
                }else if (skipT > 0){
                    skipT--;
                    j--;
                }else {
                    break;
                }
            }
            if (i >= 0 && j >= 0){
                if (s.charAt(i) != t.charAt(j)){
                    return false;
                }
            }else if (i >= 0 || j >= 0){
                return false;
            }
            i--;
            j--;
        }
        return true;
    }

    /**
     * 876. 链表的中间结点
     * 给定一个头结点为 head 的非空单链表，返回链表的中间结点。
     *
     * 如果有两个中间结点，则返回第二个中间结点。
     */
    public ListNode middleNode(ListNode head) {
        ListNode p = head;
        ListNode q = head;
        while (q!= null && q.next != null){
            q = q.next;
            q = q.next;
            p = p.next;
        }
        return p;
    }

    /**
     * 883 : 三维形体投影面积
     * 在n x n的网格grid中，我们放置了一些与 x，y，z 三轴对齐的1 x 1 x 1立方体。
     * <p>
     * 每个值v = grid[i][j]表示 v个正方体叠放在单元格(i, j)上。
     * <p>
     * 现在，我们查看这些立方体在 xy、yz和 zx平面上的投影。
     * <p>
     * 投影就像影子，将 三维 形体映射到一个 二维 平面上。从顶部、前面和侧面看立方体时，我们会看到“影子”。
     * <p>
     * 返回 所有三个投影的总面积 。
     */
    public int projectionArea(int[][] grid) {
        int xy = 0;
        int yz = 0;
        int xz = 0;
        int maxYZ;
        int[] maxXZ = new int[grid[0].length];
        for (int i = 0; i < grid.length; i++) {
            maxYZ = 0;
            for (int j = 0; j < grid[0].length; j++) {
                xy += grid[i][j] == 0 ? 0 : 1;
                maxYZ = Math.max(maxYZ, grid[i][j]);
                maxXZ[j] = Math.max(maxXZ[j], grid[i][j]);
            }
            yz += maxYZ;
        }
        xz = Arrays.stream(maxXZ).sum();
        int area = xy + xz + yz;
        return area;
    }

    /**
     * 942. 增减字符串匹配
     * 由范围 [0,n] 内所有整数组成的 n + 1 个整数的排列序列可以表示为长度为 n 的字符串 s ，其中:
     *
     * 如果perm[i] < perm[i + 1]，那么s[i] == 'I'
     * 如果perm[i] > perm[i + 1]，那么 s[i] == 'D'
     * 给定一个字符串 s ，重构排列perm 并返回它。如果有多个有效排列perm，则返回其中 任何一个 。
     */
    public int[] diStringMatch(String s) {
        int n = s.length(), lo = 0, hi = n;
        int[] perm = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            perm[i] = s.charAt(i) == 'I' ? lo++ : hi--;
        }
        perm[n] = lo; // 最后剩下一个数，此时 lo == hi
        return perm;
    }

    /**
     * 961. 在长度 2N 的数组中找出重复 N 次的元素
     * 给你一个整数数组 nums ，该数组具有以下属性：
     *
     * nums.length == 2 * n.
     * nums 包含 n + 1 个 不同的 元素
     * nums 中恰有一个元素重复 n 次
     * 找出并返回重复了 n 次的那个元素。
     */
    public int repeatedNTimes(int[] nums) {
        Set<Integer> set = new HashSet<>();
        int ans = 0;
        for (int num : nums) {
            if (set.contains(num)){
                ans = num;
                break;
            }else {
                set.add(num);
            }
        }
        return ans;
    }

    /**
     * 965. 单值二叉树
     * 如果二叉树每个节点都具有相同的值，那么该二叉树就是单值二叉树。
     *
     * 只有给定的树是单值二叉树时，才返回 true；否则返回 false。
     */
    public boolean isUnivalTree(TreeNode root) {
        if (root == null){
            return true;
        }
        return isUnivalTreeHelper(root, root.val);
    }

    private boolean isUnivalTreeHelper(TreeNode root, int val) {
        if (root == null){
            return true;
        }
        if (root.val != val){
            return false;
        }
        return isUnivalTreeHelper(root.left, val) && isUnivalTreeHelper(root.right, val);
    }

    /**
     * 977. 有序数组的平方
     * 给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。
     */
    public int[] sortedSquares(int[] nums) {
        int[] ans = new int[nums.length];
        int i = 0;
        int j = nums.length - 1;
        int k = 0;
        while (i <= j){
            int a = nums[i] * nums[i];
            int b = nums[j] * nums[j];
            if (a > b){
                ans[ans.length - k - 1] = a;
                i++;
            }else {
                ans[ans.length - k - 1] = b;
                j--;
            }
            k++;
        }
        return ans;
    }

    /**
     * 986: 区间列表的交集
     * 给定两个由一些 闭区间 组成的列表，firstList 和 secondList ，其中 firstList[i] = [starti, endi] 而 secondList[j] = [startj, endj] 。每个区间列表都是成对 不相交 的，并且 已经排序 。
     * <p>
     * 返回这 两个区间列表的交集 。
     * <p>
     * 形式上，闭区间 [a, b]（其中 a <= b）表示实数 x 的集合，而 a <= x <= b 。
     * <p>
     * 两个闭区间的 交集 是一组实数，要么为空集，要么为闭区间。例如，[1, 3] 和 [2, 4] 的交集为 [2, 3] 。
     */
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        List<int[]> ans = new ArrayList<>();
        int i = 0;
        int j = 0;
        while (i < firstList.length && j < secondList.length) {
            int a = Math.max(firstList[i][0], secondList[j][0]);
            int b = Math.min(firstList[i][1], secondList[j][1]);
            if (a <= b) {
                ans.add(new int[]{a, b});
            }
            if (firstList[i][1] < secondList[j][1]) {
                i++;
            } else {
                j++;
            }
        }
        return ans.toArray(new int[ans.size()][]);
    }

    public int[][] intervalIntersection2(int[][] firstList, int[][] secondList) {
        List<int[]> list = new ArrayList<>();
        for (int[] one : firstList) {
            for (int[] two : secondList) {
                if (two[0] <= one[1] && two[1] >= one[0]){
                    int[] temp = new int[2];
                    temp[0] = Math.max(one[0], two[0]);
                    temp[1] = Math.min(one[1], two[1]);
                    list.add(temp);
                }
            }
        }
        return list.toArray(new int[list.size()][]);
    }

    /**
     * 994. 腐烂的橘子
     * 在给定的m x n网格grid中，每个单元格可以有以下三个值之一：
     *
     * 值0代表空单元格；
     * 值1代表新鲜橘子；
     * 值2代表腐烂的橘子。
     * 每分钟，腐烂的橘子周围4 个方向上相邻 的新鲜橘子都会腐烂。
     *
     * 返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回-1。
     */
    public int orangesRotting(int[][] grid) {
        Queue<int[]> queue = new LinkedList<>();
        int oranges = 0;
        for (int row = 0; row < grid.length; row++) {
            for (int col = 0; col < grid[0].length; col++) {
                if (grid[row][col] == 2){
                    queue.offer(new int[]{row, col});
                }else if (grid[row][col] == 1){
                    oranges++;
                }
            }
        }
        int[] directions = {-1, 0, 1, 0, -1};
        int time = -1;
        while (!queue.isEmpty()){
            int size = queue.size();
            while (size-- > 0){
                int[] cur = queue.poll();
                for (int i = 0; i < directions.length - 1; i++) {
                    int x = cur[0] + directions[i];
                    int y = cur[1] + directions[i + 1];
                    if (x < 0 || x >= grid.length || y < 0 || y >= grid[0].length || grid[x][y] == 0 || grid[x][y] == 2){
                        continue;
                    }
                    grid[x][y] = 2;
                    oranges--;
                    queue.offer(new int[]{x, y});
                }
            }
            time++;
        }
        return oranges == 0 ? Math.max(time, 0) : -1;
    }

    /**
     * 1021. 删除最外层的括号
     * 有效括号字符串为空 ""、"(" + A + ")"或A + B ，其中A 和B都是有效的括号字符串，+代表字符串的连接。
     *
     * 例如，""，"()"，"(())()"和"(()(()))"都是有效的括号字符串。
     * 如果有效字符串 s 非空，且不存在将其拆分为 s = A + B的方法，我们称其为原语（primitive），其中A 和B都是非空有效括号字符串。
     *
     * 给出一个非空有效字符串 s，考虑将其进行原语化分解，使得：s = P_1 + P_2 + ... + P_k，其中P_i是有效括号字符串原语。
     *
     * 对 s 进行原语化分解，删除分解中每个原语字符串的最外层括号，返回 s 。
     */
    public String removeOuterParentheses(String s) {
        Stack<Character> stack = new Stack<>();
        StringBuilder res = new StringBuilder();
        char[] chars = s.toCharArray();
        for (char c : chars) {
            if (c == '('){
                stack.push(c);
                if (stack.size() > 1){
                    res.append(c);
                }
            }else if (c == ')'){
                if (stack.size() > 1){
                    res.append(c);
                }
                stack.pop();
            }
        }
        return res.toString();
    }

    /**
     * 1047. 删除字符串中的所有相邻重复项
     * 给出由小写字母组成的字符串S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。
     *
     * 在 S 上反复执行重复项删除操作，直到无法继续删除。
     *
     * 在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。
     */
    public String removeDuplicates(String s) {
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (!stack.isEmpty() && c == stack.peek()){
                stack.pop();
            }else {
                stack.push(c);
            }
        }
        char[] chars = new char[stack.size()];
        for (int i = chars.length - 1; i >= 0; i--) {
            chars[i] = stack.pop();
        }
        return new String(chars);
    }

    /**
     * 1049. 最后一块石头的重量 II
     * 有一堆石头，用整数数组stones 表示。其中stones[i] 表示第 i 块石头的重量。
     *
     * 每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为x 和y，且x <= y。那么粉碎的可能结果如下：
     *
     * 如果x == y，那么两块石头都会被完全粉碎；
     * 如果x != y，那么重量为x的石头将会完全粉碎，而重量为y的石头新重量为y-x。
     * 最后，最多只会剩下一块 石头。返回此石头 最小的可能重量 。如果没有石头剩下，就返回 0。
     */
    public int lastStoneWeightII(int[] stones) {
        int weight = 0;
        for (int stone : stones) {
            weight += stone;
        }
        int target = weight / 2;
        int[] dp = new int[target + 1];
        for (int i = 0; i < stones.length; i++) {
            for (int j = target; j >= stones[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - stones[i]] + stones[i]);
            }
        }
        return weight - dp[target] - dp[target];
    }

    /**
     * 1091. 二进制矩阵中的最短路径
     * 给你一个 n x n 的二进制矩阵 grid 中，返回矩阵中最短 畅通路径 的长度。如果不存在这样的路径，返回 -1 。
     *
     * 二进制矩阵中的 畅通路径 是一条从 左上角 单元格（即，(0, 0)）到 右下角 单元格（即，(n - 1, n - 1)）的路径，该路径同时满足下述要求：
     *
     * 路径途经的所有单元格都的值都是 0 。
     * 路径中所有相邻的单元格应当在 8 个方向之一 上连通（即，相邻两单元之间彼此不同且共享一条边或者一个角）。
     * 畅通路径的长度 是该路径途经的单元格总数。
     */
    public int shortestPathBinaryMatrix(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0] == null || grid[0].length == 0 || grid[0][0] == 1){
            return -1;
        }
        int path = 0;
        int[] directions = {1, 0, -1, 0, 1, -1, -1, 1, 1};
        Queue<int[]> queue = new LinkedList<>();
        int row, col;
        queue.offer(new int[]{0, 0});
        while (!queue.isEmpty()){
            int len = queue.size();
            for (int j = 0; j < len; j++) {
                int[] node = queue.poll();
                if (node[0] == grid.length - 1 && node[1] == grid[0].length - 1){
                    return path + 1;
                }
                row = node[0];
                col = node[1];
                grid[row][col] = 2;
                for (int i = 0; i < directions.length - 1; i++) {
                    row += directions[i];
                    col += directions[i + 1];
                    if (row < 0 || col < 0 || row == grid.length || col == grid[0].length || grid[row][col] != 0){
                    }else {
                        queue.offer(new int[]{row, col});
                        grid[row][col] = 1;
                    }
                    row -= directions[i];
                    col -= directions[i + 1];
                }
            }
            path++;
        }
        return -1;
    }

    /**
     * 1143. 最长公共子序列
     * 给定两个字符串text1 和text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
     *
     * 一个字符串的子序列是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
     *
     * 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
     * 两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。
     */
    public int longestCommonSubsequence(String text1, String text2) {
        int M = text1.length();
        int N = text2.length();
        int[][] dp = new int[M + 1][N + 1];
        for (int i = 1; i <= M; i++) {
            for (int j = 1; j <= N; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[M][N];
    }

    /**
     * 1305. 两棵二叉搜索树中的所有元素
     * 给你 root1 和 root2 这两棵二叉搜索树。请你返回一个列表，其中包含 两棵树 中的所有整数并按 升序 排序。
     */
    public List<Integer> getAllElements(TreeNode root1, TreeNode root2) {
        Queue<Integer> q1 = new LinkedList<>();
        Queue<Integer> q2 = new LinkedList<>();
        inorder(root1, q1);
        inorder(root2, q2);
        List<Integer> ans = new ArrayList<>();
        while (!q1.isEmpty() && !q2.isEmpty()){
            if (q1.peek() < q2.peek()){
                ans.add(q1.poll());
            }else {
                ans.add(q2.poll());
            }
        }
        while (!q1.isEmpty()){
            ans.add(q1.poll());
        }
        while (!q2.isEmpty()){
            ans.add(q2.poll());
        }
        return ans;
    }
    private void inorder(TreeNode node, Queue<Integer> q){
        if (node != null){
            inorder(node.left, q);
            q.offer(node.val);
            inorder(node.right, q);
        }
    }


    /**
     * 背包01
     */
    public static void testWeightBagProblem(int[] weight, int[] value, int bagSize){
        //定义dp数组，dp[i][j]表示背包容量为j时，前i个物品能获得的最大价值
        int[][] dp = new int[weight.length][bagSize + 1];
        //初始化   背包容量为0时，获得的价值均为0
        for (int i = 0; i < weight.length; i++) {
            dp[i][0] = 0;
        }
        //初始化   当只有第0件商品时，价值都为valur[0]
        for (int j = 1; j <= bagSize; j++) {
            dp[0][j] = value[0];
        }
        //先遍历物品，再遍历背包
        for (int i = 1; i < weight.length; i++) {
            for (int j = 1; j <= bagSize; j++) {
                if (j < weight[i]){
                    dp[i][j] = dp[i - 1][j];
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);
                }
            }
        }
        for (int[] ints : dp) {
            for (int anInt : ints) {
                System.out.print(anInt + ",");
            }
            System.out.println("");
        }
    }

    //一维数组
    public static void testWeightBagProblem2(int[] weight, int[] value, int bagSize){
        int[] dp = new int[bagSize + 1];
        for (int i = 0; i < weight.length; i++) {
            for (int j = bagSize; j >= weight[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - weight[i]] + value[i]);
            }
        }

        //打印dp数组
        for (int i = 0; i <= bagSize; i++) {
            System.out.println(dp[i] + "");
        }
    }

    /**
     * 面试题 02.07. 链表相交
     * 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。
     */
    public ListNode getIntersectionNode2(ListNode headA, ListNode headB) {
        if (headA == null || headB == null){
            return null;
        }
        ListNode p = headA;
        ListNode q = headB;
        int a = 0;
        int b = 0;
        while (p != null){
            p = p.next;
            a++;
        }
        while (q != null){
            q = q.next;
            b++;
        }
        p = headA;
        q = headB;
        while (a > b){
            p = p.next;
            a--;
        }
        while (a < b){
            q = q.next;
            b--;
        }
        while (p != null && q != null && !p.equals(q)){
            p = p.next;
            q = q.next;
        }
        if (p != null && q != null){
            return p;
        }
        return null;
    }

    //双指针
    public ListNode getIntersectionNode3(ListNode headA, ListNode headB) {
        if (headA == null || headB == null){
            return null;
        }
        ListNode a = headA;
        ListNode b = headB;
        while (a != b){
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
        }
        return a;
    }
}


/**
 * 384. 打乱数组
 * 给你一个整数数组 nums ，设计算法来打乱一个没有重复元素的数组。打乱后，数组的所有排列应该是等可能的。
 *
 * 实现 Solution class:
 *
 * Solution(int[] nums) 使用整数数组 nums 初始化对象
 * int[] reset() 重设数组到它的初始状态并返回
 * int[] shuffle() 返回数组随机打乱后的结果
 */
class Solution2 {
    int[] nums;
    int[] original;
    public Solution2(int[] nums) {
        this.nums = nums;
        this.original = Arrays.copyOf(nums, nums.length);

    }

    public int[] reset() {
        return Arrays.copyOf(original, nums.length);
    }

    //Fisher-Yates 洗牌算法
    public int[] shuffle() {
        Random random = new Random();
        for (int i = 0; i < nums.length; i++) {
            int j = i + random.nextInt(nums.length - i);
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }
        return nums;
    }
}

/**
 * 232. 用栈实现队列
 * 请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：
 *
 * 实现 MyQueue 类：
 *
 * void push(int x) 将元素 x 推到队列的末尾
 * int pop() 从队列的开头移除并返回元素
 * int peek() 返回队列开头的元素
 * boolean empty() 如果队列为空，返回 true ；否则，返回 false
 * 说明：
 *
 * 你 只能 使用标准的栈操作 —— 也就是只有push to top,peek/pop from top,size, 和is empty操作是合法的。
 * 你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。
 */
class MyQueue {

    Deque<Integer> deque;
    public MyQueue() {
        deque = new ArrayDeque<>();
    }

    public void push(int x) {
        deque.offer(x);
    }

    public int pop() {
        return deque.pollFirst();
    }

    public int peek() {
        return deque.peekFirst();
    }

    public boolean empty() {
        return deque.isEmpty();
    }
}

class MyQueue2 {

    Stack<Integer> stackIn;
    Stack<Integer> stackOut;
    public MyQueue2() {
        stackIn = new Stack<>();
        stackOut = new Stack<>();
    }

    public void push(int x) {
        stackIn.push(x);
    }

    public int pop() {
        if (stackOut.isEmpty()){
            while (!stackIn.isEmpty()){
                stackOut.push(stackIn.pop());
            }
        }
        return stackOut.pop();
    }

    public int peek() {
        if (stackOut.isEmpty()){
            while (!stackIn.isEmpty()){
                stackOut.push(stackIn.pop());
            }
        }
        return stackOut.peek();
    }

    public boolean empty() {
        return stackIn.isEmpty() && stackOut.isEmpty();
    }
}

/**
 * 225. 用队列实现栈
 */
class MyStack {
    Queue<Integer> queueIn;
    public MyStack() {
        queueIn = new LinkedList<>();
    }

    public void push(int x) {
        queueIn.offer(x);
    }

    public int pop() {
        int len = queueIn.size();
        while (len > 1){
            queueIn.offer(queueIn.poll());
            len--;
        }
        return queueIn.poll();
    }

    public int top() {
        int len = queueIn.size();
        while (len > 1){
            queueIn.offer(queueIn.poll());
            len--;
        }
        int res = queueIn.poll();
        queueIn.offer(res);
        return res;
    }

    public boolean empty() {
        return queueIn.isEmpty();
    }
}

class MyOrderQueue {
    Deque<Integer> deque = new LinkedList<>();
    void poll(int val){
        if (!deque.isEmpty() && val == deque.peek()){
            deque.pop();
        }
    }

    void add(int val){
        while (!deque.isEmpty() && val > deque.getLast()){
            deque.pollLast();
        }
        deque.add(val);
    }

    int peek(){
        return deque.peek();
    }
}

