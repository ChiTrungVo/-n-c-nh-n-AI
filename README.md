1. Mục tiêu
  Xây dựng chương trình giải bài toán 8 ô số bằng cách cài đặt và so sánh hiệu quả của các thuật toán tìm kiếm khác nhau.    
  Trực quan hóa quá trình giải bài toán bằng giao diện đồ họa (GUI) sử dụng thư viện Pygame.
  Phân tích ưu và nhược điểm của từng thuật toán khi áp dụng vào bài toán cụ thể này.

2. Nội dung
   
  2.1. Các thuật toán tìm kiếm không có thông tin
    Các thành phần chính của bài toán tìm kiếm:
    Trạng thái (State): Một bảng 8 ô số.
    Hành động (Action): Các phép di chuyển ô trống (lên, xuống, trái, phải).
    Trạng thái ban đầu (Initial State): Cấu hình ban đầu của bảng.
    Trạng thái mục tiêu (Goal State): Cấu hình đích mà ta muốn đạt được.
    Chi phí đường đi (Path Cost): Số bước di chuyển để đạt đến trạng thái mục tiêu.    
    Solution: Một dãy các hành động dẫn từ trạng thái ban đầu đến trạng thái mục tiêu.    
    Các thuật toán đã cài đặt:
      BFS (Tìm kiếm theo chiều rộng)
      DFS (Tìm kiếm theo chiều sâu)
      UCS (Tìm kiếm chi phí đồng nhất)
      IDS (Tìm kiếm sâu dần)
  
  2.2. Các thuật toán tìm kiếm có thông tin
    Các thành phần chính của bài toán tìm kiếm:
      Trạng thái (State): Một bảng 8 ô số.
      Hành động (Action): Các phép di chuyển ô trống (lên, xuống, trái, phải).
      Trạng thái ban đầu (Initial State): Cấu hình ban đầu của bảng.
      Trạng thái mục tiêu (Goal State): Cấu hình đích mà ta muốn đạt được.
      Chi phí đường đi (Path Cost): Số bước di chuyển để đạt đến trạng thái mục tiêu.    
      Solution: Một dãy các hành động dẫn từ trạng thái ban đầu đến trạng thái mục tiêu.    
    Các thuật toán đã cài đặt:
      Greedy Best-First Search
      A* Search
      IDA* Search
  
  2.3. Local search
    Các thành phần chính của bài toán tìm kiếm:
      Trạng thái (State): Một bảng 8 ô số.
      Hành động (Action): Các phép di chuyển ô trống (lên, xuống, trái, phải).
      Trạng thái ban đầu (Initial State): Cấu hình ban đầu của bảng.
      Trạng thái mục tiêu (Goal State): Cấu hình đích mà ta muốn đạt được.
      Chi phí đường đi (Path Cost): Số bước di chuyển để đạt đến trạng thái mục tiêu.    
      Solution: Một dãy các hành động dẫn từ trạng thái ban đầu đến trạng thái mục tiêu.    
    Các thuật toán đã cài đặt:
      Hill Climbing (Simple, Steepest Ascent, Stochastic)
      Simulated Annealing
      Genetic Algorithm
      Beam Search
  
  2.4. Tìm kiếm trong môi trường phức tạp
    Giới thiệu:
      Trong thực tế, nhiều bài toán tìm kiếm diễn ra trong các môi trường phức tạp hơn so với môi trường tĩnh và quan sát được hoàn toàn của bài toán 8 ô số.
      Các yếu tố phức tạp bao gồm:
        Môi trường không xác định (không biết trước kết quả của hành động).
        Môi trường quan sát được một phần (chỉ nhận được một phần thông tin về trạng thái).
        Môi trường động (trạng thái môi trường có thể thay đổi).
    Các thuật toán đã cài đặt:
     Tree Search AND – OR
     Partially Observable  ( nhìn thấy một phần)
     Unknown or Dynamic Environment  ( Không nhìn thấy hoàn toàn – tìm kiếm trong môi trường niềm tin)
  
  2.5. Các thuật toán tìm kiếm có ràng buộc
    Các thành phần của bài toán:
      Biến (Variable): Mỗi ô trên bảng 8 ô số có thể được xem là một biến.
      Miền giá trị (Domain): Tập hợp các giá trị có thể gán cho một biến (các số từ 0 đến 8).
      Ràng buộc (Constraint): Các ràng buộc đảm bảo tính hợp lệ của trạng thái (ví dụ: không có hai ô nào có cùng giá trị, ô trống chỉ có thể di chuyển sang ô kề cạnh).
    Các thuật toán đã cài đặt:
      Backtracking Search
      Forward Checking
      AC-3
  
  2.6. Reinforcement Learning
    Giới thiệu:
      Học tăng cường là một nhánh của trí tuệ nhân tạo cho phép agent học cách hành động tối ưu trong một môi trường bằng cách nhận phần thưởng hoặc hình phạt.
      Mặc dù các thuật toán tìm kiếm truyền thống tập trung vào việc tìm đường đi từ trạng thái ban đầu đến trạng thái mục tiêu, học tăng cường có thể được sử dụng để huấn luyện một agent giải bài toán 8 ô số một cách hiệu quả, đặc biệt trong các biến thể phức tạp hơn của bài toán.
    Các thuật toán cài đặt:
      Q-Learning
      Temporal Difference (TD) Learning
