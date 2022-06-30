package MyClassDemo;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

public class InitDemo {
    /**
     *创建一条随机String
     */
    public String createString(int length){
        CreateOne cc = new CreateNums();
        String s = cc.createRandomString(length);
        System.out.println(s);
        return s;
    }


    /**
     *创建一条随机链表
     */
    public ListNode createListNode(int length){
        ListNode headNode = new ListNode();
        headNode.next = null;
        Random r = new Random();
        ListNode l = headNode.next;

        for (int i = 0; i < length; i++) {
            ListNode p = new ListNode();
            p.val = r.nextInt(10);
            p.next = l;
            headNode.next = p;
            l = p;
        }
        return headNode;
    }


    /**
     * 顺序打印带头结点单链表
     * @param head 头结点
     */
    public void orderedPrintLinkListWithHead(ListNode head){
        //顺序打印单链表
        StringBuilder linkList = new StringBuilder();
        linkList.append("head");

        ListNode p = head;
        while (p.next != null){
            p = p.next;
            linkList.append("-->");
            linkList.append("[" + p.val + "]");
        }
        System.out.println(linkList);
    }

    /**
     * 顺序打印不带头结点单链表
     * @param head
     */
    public void orderedPrintLinkListWithoutHead(ListNode head){
        //顺序打印单链表
        StringBuilder linkList = new StringBuilder();
        linkList.append("[" + head.val + "]");

        ListNode p = head;
        while (p.next != null){
            p = p.next;
            linkList.append("-->");
            linkList.append("[" + p.val + "]");
        }
        System.out.println(linkList);
    }

    //数组往前移动n位
    public void removeNSums(int[] nums, int tmp){
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, tmp - 1);
        reverse(nums, tmp , nums.length - 1);
    }




    //数组的逆置
    public void reverse(int[] nums,int begin, int end){
        int a = begin;
        int b = end;
        int emp;
        for (int i = 0; i < (b - a + 1)/2; i++){
            emp = nums[a + i];
            nums[a + i] = nums[b - i];
            nums[b - i] = emp;
        }
    }


    //根据数组生成二叉树
    public TreeNode convert(Integer[] array) {
        int len = array.length;
        TreeNode root = new TreeNode();
        root.val = array[0];
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int i = 1;
        while (!queue.isEmpty()){
            int size = queue.size();
            for (int j = 0; j < size; j++) {
                TreeNode node = queue.poll();
                int index = i * 2 - 1;
                if (index < len && array[index] != null){
                    TreeNode left = new TreeNode();
                    left.val = array[index];
                    node.left = left;
                    queue.offer(left);
                }else {
                    node.left = null;
                }
                index++;
                if (index < len && array[index] != null){
                    TreeNode right = new TreeNode();
                    right.val = array[index];
                    node.right = right;
                    queue.offer(right);
                }else {
                    node.right = null;
                }
            }
        }
        return root;
    }

    //根据数组创建链表
    public ListNode createLinkedList(int[] nums){
        ListNode head = new ListNode();
        head.next = null;
        ListNode n = head;
        for (int num : nums) {
            ListNode p = new ListNode();
            p.next = n.next;
            n.next = p;
            p.val = num;
            n = p;
        }
        return head.next;
    }
}
