import sys

def calculate_average():
    """
    从命令行参数获取数值，计算并输出它们的平均值。
    """
    # sys.argv[0] 是脚本本身的名称，所以我们从 sys.argv[1] 开始获取实际参数
    args = sys.argv[1:]

    if not args:
        print("用法: python your_script_name.py <数值1> <数值2> ...")
        print("例如: python your_script_name.py 10 20 30 40")
        return

    numbers = []
    for arg in args:
        try:
            # 尝试将每个参数转换为浮点数
            number = float(arg)
            numbers.append(number)
        except ValueError:
            print(f"错误: '{arg}' 不是一个有效的数值。请确保所有输入都是数字。")
            return

    if not numbers:
        print("没有有效的数值可以计算平均值。")
        return

    total_sum = sum(numbers)
    count = len(numbers)
    average = total_sum / count

    print(f"输入的数值: {numbers}")
    print(f"总和: {total_sum}")
    print(f"数量: {count}")
    print(f"平均值: {average}")

if __name__ == "__main__":
    calculate_average()