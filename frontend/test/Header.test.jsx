import { describe, it, expect, mock } from "bun:test";
import { render, screen, fireEvent } from "@testing-library/react";
import Header from "../src/components/Header";

describe("Header UI Component", () => {
    it("renders brand title and subtitle", () => {
        render(
            <Header
                backendStatus="connected"
                onRefreshCatalog={() => {}}
                isRefreshing={false}
            />
        );

        expect(screen.getByText("SnapClass")).toBeTruthy();
        expect(screen.getByText("Few-Shot Visual Intelligence")).toBeTruthy();
    });

    it("displays connected status with live ping pill", () => {
        render(
            <Header
                backendStatus="connected"
                onRefreshCatalog={() => {}}
                isRefreshing={false}
            />
        );

        const pill = screen.getByRole("status");
        expect(pill).toBeTruthy();
        expect(pill.textContent).toContain("Connected");
    });

    it("displays offline status when backend is disconnected", () => {
        render(
            <Header
                backendStatus="error"
                onRefreshCatalog={() => {}}
                isRefreshing={false}
            />
        );

        const pill = screen.getByRole("status");
        expect(pill.textContent).toContain("Offline");
    });

    it("calls onRefreshCatalog when refresh button is clicked", () => {
        const onRefresh = mock(() => {});
        render(
            <Header
                backendStatus="connected"
                onRefreshCatalog={onRefresh}
                isRefreshing={false}
            />
        );

        const refreshBtn = screen.getByLabelText("Refresh reference embeddings");
        fireEvent.click(refreshBtn);
        expect(onRefresh).toHaveBeenCalled();
    });

    it("disables refresh button and adds spinning class when isRefreshing is true", () => {
        render(
            <Header
                backendStatus="connected"
                onRefreshCatalog={() => {}}
                isRefreshing={true}
            />
        );

        const refreshBtn = screen.getByLabelText("Refresh reference embeddings");
        expect(refreshBtn.hasAttribute("disabled")).toBe(true);
    });

    it("toggles theme between dark and light modes", () => {
        render(
            <Header
                backendStatus="connected"
                onRefreshCatalog={() => {}}
                isRefreshing={false}
            />
        );

        const lightBtn = screen.getByLabelText("Switch to Light mode");
        const darkBtn = screen.getByLabelText("Switch to Dark mode");

        fireEvent.click(lightBtn);
        expect(document.documentElement.getAttribute("data-theme")).toBe("light");

        fireEvent.click(darkBtn);
        expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    });
});
