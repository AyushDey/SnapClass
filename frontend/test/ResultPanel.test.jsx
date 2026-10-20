import { describe, it, expect } from "bun:test";
import { render, screen } from "@testing-library/react";
import ResultPanel from "../src/components/ResultPanel";

describe("ResultPanel UI Component", () => {
    it("renders empty state placeholder when no result is present", () => {
        render(<ResultPanel result={null} isClassifying={false} />);

        expect(screen.getByText("Classification Results")).toBeTruthy();
        expect(screen.getByText("No Image Loaded")).toBeTruthy();
        expect(
            screen.getByText("Capture a photo or upload an image file to compare against reference classes.")
        ).toBeTruthy();
    });

    it("renders skeleton loader when isClassifying is true", () => {
        render(<ResultPanel result={null} isClassifying={true} />);

        expect(screen.getByText("Classification Results")).toBeTruthy();
        expect(screen.getByText("Computing visual feature embeddings…")).toBeTruthy();
        const skeletons = document.querySelectorAll(".skeleton-pulse");
        expect(skeletons.length).toBeGreaterThan(0);
    });

    it("renders match results when classification succeeds", () => {
        const result = {
            class: "Boeing_777",
            confidence: 0.942,
            category_name: "Aviation",
            matches: [
                { class: "Boeing_747", score: 0.811, category_name: "Aviation" },
                { class: "Airbus_A380", score: 0.725, category_name: "Aviation" },
            ],
        };

        render(<ResultPanel result={result} isClassifying={false} />);

        expect(screen.getByText("Boeing 777")).toBeTruthy();
        expect(screen.getByText("94.2%")).toBeTruthy();
        expect(screen.getByText("Category Candidates (2)")).toBeTruthy();
        expect(screen.getByText("Boeing 747")).toBeTruthy();
        expect(screen.getByText("Airbus A380")).toBeTruthy();
    });
});
