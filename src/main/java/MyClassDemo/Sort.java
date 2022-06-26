package MyClassDemo;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class Sort {
    public void mySort(int[][] array){
        List<Integer> list = new ArrayList<>();
        for (int[] ints : array) {
            for (int anInt : ints) {
                list.add(anInt);
            }
        }
        list.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o1 - o2;
            }
        });
        int n = 0;
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j< array[i].length; j++) {
                array[i][j] = list.get(n++);
            }
        }
    }

    //冒泡排序
    public void bubbleSort(int[] nums){
        boolean hasChange = true;
        for (int i = 0; i < nums.length && hasChange; i++) {
            hasChange = false;
            for (int j = 0; j < nums.length - 1 - i; j++) {
                if (nums[j] > nums[j + 1]){
                    swap(nums, j, j + 1);
                    hasChange = true;
                }
            }
        }
    }
    private void swap(int[] nums, int a, int b){
        int temp = nums[a];
        nums[a] = nums[b];
        nums[b] = temp;
    }

    //插入排序
    public void insertSort(int[] nums){
        for (int i = 1, j; i < nums.length; i++) {
            int current = nums[i];
            for (j = i - 1; j >= 0 && nums[j] < current; j++) {
                nums[j + 1] = nums[j];
            }
            nums[j + 1] = nums[i];
        }
    }

    //归并排序
    public void mergerSort(int[] nums, int left, int right){
        if (left >= right){
            return;
        }
        int mid = (left + right) / 2;
        mergerSort(nums, left, mid);
        mergerSort(nums, mid + 1, right);
        merge(nums, left, mid, right);
    }

    private void merge(int[] nums, int left, int mid, int right) {
        int[] copy = nums.clone();
        int k = left;
        int i = left;
        int j = mid + 1;
        while (k <= right){
            if (i > mid){
                nums[k++] = copy[j++];
            }else if (j > right){
                nums[k++] = copy[i++];
            }else if (nums[i] < nums[j]){
                nums[k++] = nums[i++];
            }else {
                nums[k++] = nums[j++];
            }
        }
    }

    //快速排序
    public void quickSort(int[] nums, int low, int high){
        if (low < high) {
            //哨兵划分操作
            int p = partition(nums, low, high);
            quickSort(nums, low, p - 1);
            quickSort(nums, p + 1, high);
        }
    }

    private int partition(int[] nums, int low, int high) {
        swap(nums, (int) (Math.random() * (high - low + 1) + low), high);
        int i, j;
        for (i = low, j = low; j < high; j++) {
            if (nums[j] <= nums[high]){
                swap(nums, i++, j);
            }
        }
        swap(nums, i, j);
        return i;
    }
}
