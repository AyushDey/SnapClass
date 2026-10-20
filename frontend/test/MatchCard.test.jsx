import { describe, it, expect } from "bun:test";
import { render, screen } from "@testing-library/react";
import MatchCard from "../src/components/MatchCard";

describe("MatchCard UI Component", () => {
    it("renders top rank-1 best match card with score and badge", () => {
        const match = {
            label: "flower_vase",
            score: 0.965,
        };

        render(<MatchCard match={match} rank={1} isTopMatch={true} />);

        expect(screen.getByText("flower vase")).toBeTruthy();
        expect(screen.getByText("96.5%")).toBeTruthy();
        expect(screen.getByText("Top Match")).toBeTruthy();
    });

    it("renders runner-up match card with rank number and progress fill", () => {
        const match = {
            label: "table_lamp",
            score: 0.824,
        };

        render(<MatchCard match={match} rank={2} isTopMatch={false} />);

        expect(screen.getByText("table lamp")).toBeTruthy();
        expect(screen.getByText("82.4%")).toBeTruthy();
        expect(screen.getByText("#2")).toBeTruthy();
    });

    it("handles zero or null score gracefully", () => {
        const match = {
            label: "unknown_object",
            score: 0,
        };

        render(<MatchCard match={match} rank={3} isTopMatch={false} />);

        expect(screen.getByText("unknown object")).toBeTruthy();
        expect(screen.getByText("0.0%")).toBeTruthy();
    });
});
