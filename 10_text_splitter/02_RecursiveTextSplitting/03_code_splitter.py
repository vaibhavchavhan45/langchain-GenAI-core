from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = '''
    public class _02_length {

    //Finding length of LL using iterative method
    public static int length(Node head){
        int count = 0;
        while(head != null){
            count++;
            head = head.next;
        }
        return count;
    }

    //Finding length of LL using recursive method
    public static int length_r(Node head){
        if (head == null) return 0;
        return 1 + length_r(head.next);
    }

    public static class Node{
        int data;
        Node next;
        Node(int data){
            this.data = data;
        }
    }

    public static void main(String[] args) {
        Node a = new Node(1);
        Node b = new Node(1);
        Node c = new Node(1);
        Node d = new Node(1);
        Node e = new Node(1);

        a.next = b;
        b.next = c;
        c.next = d;
        d.next = e;

        System.out.println(length(a));
        System.out.println(length_r(a));
    }
}

'''
splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.JAVA,
    chunk_size = 500,
    chunk_overlap = 0
)

chunk = splitter.split_text(code)

print(len(chunk))
print(chunk[1])