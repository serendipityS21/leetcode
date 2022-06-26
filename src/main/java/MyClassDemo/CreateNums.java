package MyClassDemo;

import java.util.Random;

public class CreateNums implements CreateOne {







    /**
     * 创建静态数组
     * @param nums 数组
     */
    public void createStaticArrayList(int[] nums) {
        nums = new int[]{16, 5, 9, 9, 18, 8, 0, 2, 3, 16, 5, 19, 13, 15, 18, 9, 5, 5, 12, 0};
    }


    /**
     * 创建静态数组
     * @param nums 数组
     */
    public void createRamadanArrayList(int[] nums) {
        for (int i = 0; i < 20; i++) {
            nums[i] = ((int)(Math.random() * 20));
        }
    }


    /**
     * 创建动态数组
     * @param nums 数组
     * @param length 长度
     */
    public void createRamadanArrayList(int[] nums, int length) {
        for (int i = 0; i < length; i++) {
            nums[i] = ((int)(Math.random() * length));
        }
    }

    @Override
    public void createRamadan2DArray(int[][] matrix, int length) {
        Random random = new Random();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j< matrix[i].length; j++) {
                matrix[i][j] = random.nextInt(100);
            }
        }

        //排序
        //重写ArraySort方法
        Sort s = new Sort();
        s.mySort(matrix);
    }
    @Override
    public String createRandomString(int length){
        Random r = new Random();

        //将a~z, A~Z, space 加入到一个String里
        StringBuffer sb = new StringBuffer();
        for (int i = 97; i <= 122; i++) {
            sb.append((char) i);
        }
        for (int i = 65; i <= 90; i++) {
            sb.append((char) i);
        }
        sb.append((char) 32);

        int a = 0;
        StringBuilder sb2 = new StringBuilder();
        for (int i = 0; i < length; i++) {
             a = r.nextInt(5);
             if(a == 1){
                 sb2.append((char)32);
            }else {
                 sb2.append(sb.charAt(r.nextInt(sb.length())));
             }

        }
        String s = sb2.toString();
        return s;
    }


}
