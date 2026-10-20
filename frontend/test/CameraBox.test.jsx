import { describe, it, expect, mock } from "bun:test";
import { render, screen, fireEvent } from "@testing-library/react";
import CameraBox from "../src/components/CameraBox";

describe("CameraBox UI Component", () => {
    const defaultProps = {
        videoRef: { current: null },
        isActive: false,
        capturedImage: null,
        uploadedPreview: null,
        isClassifying: false,
        onClear: () => {},
        onDropFile: () => {},
    };

    it("renders empty optical dropzone when inactive and no preview", () => {
        render(<CameraBox {...defaultProps} />);

        expect(screen.getByText("Drop image here")).toBeTruthy();
        expect(screen.getByText("or use camera / upload below")).toBeTruthy();
    });

    it("renders live viewfinder badge when isActive is true", () => {
        render(<CameraBox {...defaultProps} isActive={true} />);

        expect(screen.getByText("LIVE")).toBeTruthy();
    });

    it("renders image preview and clear button when uploadedPreview is provided", () => {
        const onClear = mock(() => {});
        render(
            <CameraBox
                {...defaultProps}
                uploadedPreview="blob:http://localhost:5173/fake-preview"
                onClear={onClear}
            />
        );

        const previewImg = screen.getByAltText("Input candidate preview");
        expect(previewImg).toBeTruthy();

        const clearBtn = screen.getByLabelText("Remove image and retake");
        expect(clearBtn).toBeTruthy();

        fireEvent.click(clearBtn);
        expect(onClear).toHaveBeenCalled();
    });
});
