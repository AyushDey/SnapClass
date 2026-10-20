import { describe, it, expect, mock } from "bun:test";
import { render, screen, fireEvent } from "@testing-library/react";
import ClassifyPanel from "../src/components/ClassifyPanel";

describe("ClassifyPanel UI Component", () => {
    const defaultProps = {
        videoRef: { current: null },
        cameraActive: false,
        capturedImage: null,
        uploadedPreview: null,
        onStartCamera: () => {},
        onCapture: () => {},
        onUpload: () => {},
        onBulkUpload: () => {},
        onClassify: () => {},
        onClearImage: () => {},
        isClassifying: false,
        hasImage: false,
    };

    it("renders navigation tabs for Camera, Upload, and Catalog", () => {
        render(<ClassifyPanel {...defaultProps} />);

        expect(screen.getByRole("tab", { name: "Camera" })).toBeTruthy();
        expect(screen.getByRole("tab", { name: "Upload" })).toBeTruthy();
        expect(screen.getByRole("tab", { name: "Catalog" })).toBeTruthy();
    });

    it("disables Run Classification button when no image is loaded", () => {
        render(<ClassifyPanel {...defaultProps} hasImage={false} />);

        const classifyBtn = screen.getByRole("button", { name: "Run Classification" });
        expect(classifyBtn.hasAttribute("disabled")).toBe(true);
    });

    it("enables Run Classification button when hasImage is true", () => {
        const onClassify = mock(() => {});
        render(<ClassifyPanel {...defaultProps} hasImage={true} onClassify={onClassify} />);

        const classifyBtn = screen.getByRole("button", { name: "Run Classification" });
        expect(classifyBtn.hasAttribute("disabled")).toBe(false);

        fireEvent.click(classifyBtn);
        expect(onClassify).toHaveBeenCalled();
    });

    it("switches to Upload tab on click and displays file browse action", () => {
        render(<ClassifyPanel {...defaultProps} />);

        const uploadTab = screen.getByRole("tab", { name: "Upload" });
        fireEvent.click(uploadTab);
        expect(uploadTab.getAttribute("aria-selected")).toBe("true");
        expect(screen.getByText("Browse Image File")).toBeTruthy();
    });

    it("switches to Catalog tab on click and displays archive ingestion", () => {
        render(<ClassifyPanel {...defaultProps} />);

        const catalogTab = screen.getByRole("tab", { name: "Catalog" });
        fireEvent.click(catalogTab);
        expect(catalogTab.getAttribute("aria-selected")).toBe("true");
        expect(screen.getByText("Bulk Reference Ingestion")).toBeTruthy();
        expect(screen.getByText("Select Archive (.zip / .tar)")).toBeTruthy();
    });
});
