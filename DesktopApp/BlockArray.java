import java.util.ArrayList;

public class BlockArray {
    ArrayList<Block> verticalBlocks;
    ArrayList<Block> horizontalBlocks;

    void addVerticalNeighbours(Block block) {
        verticalBlocks.add(block);
    }

    void addHorizontalNeighbours(Block block) {
        horizontalBlocks.add(block);
    }
}