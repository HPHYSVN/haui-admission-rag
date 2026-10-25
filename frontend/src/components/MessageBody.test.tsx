import { render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { MessageBody } from "./MessageBody";

describe("MessageBody", () => {
  afterEach(() => {
    delete (window as Window & { pwned?: unknown }).pwned;
  });

  it("renders an answer containing HTML as literal text", () => {
    const payload =
      '<img src=x onerror="window.pwned=1"><script>window.pwned=1</script>';

    const { container } = render(<MessageBody content={payload} />);

    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
    expect(container.textContent).toBe(payload);
    expect((window as Window & { pwned?: unknown }).pwned).toBeUndefined();
  });

  it("renders a normal answer without altering it", () => {
    const { container } = render(
      <MessageBody content={"Học phí năm 2026 là 1.200.000đ/học phần."} />,
    );

    expect(container.textContent).toBe("Học phí năm 2026 là 1.200.000đ/học phần.");
    expect(container.querySelector(".message-body")).not.toBeNull();
  });
});
